{-# OPTIONS_GHC -Wall -fplugin Debug.Breakpoint #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Gradient Descent Based Optimization
module MLCAD23.Opt.Grad where

import           Control.Monad                   (when, (<=<), replicateM)
import           GHC.Generics
import           Data.List                       (isSuffixOf, partition)
import           Data.Default.Class
import           System.Clock
import qualified Data.Frame                as DF
import qualified Torch                     as T
import qualified Torch.Extensions          as T
import qualified Torch.Optim.CppOptim      as T

-- import           Debug.Breakpoint
import qualified MLCAD23.Config.CKT        as CKT
import qualified MLCAD23.Config.PDK        as PDK

import           MLCAD23.Opt

newtype DesignVarSpec = DesignVarSpec {operatingPoint :: Int}
newtype DesignVariable = DesignVariable [T.Parameter] deriving (Show, Generic)

toTensor :: DesignVariable -> [T.Tensor]
toTensor (DesignVariable params) = map T.toDependent params

instance T.Randomizable DesignVarSpec DesignVariable where
  sample (DesignVarSpec n) = do
      ops' <- T.makeIndependent =<< T.normalWithSizeIO 0.5 0.1 n
      pure $ DesignVariable [ops']

instance T.Parameterized DesignVariable where
  flattenParameters (DesignVariable xs) = xs

objective :: PDK.ID -> CKT.ID -> T.Tensor -> (T.Tensor -> T.Tensor)
          -> (T.Tensor -> T.Tensor)
objective pdk ckt spec mdl = obj
  where
    obj = T.sumAll . losses spec . mdl . trafoX pdk ckt . T.clamp 0.0 1.0

showLog :: (Show a) => Int -> Int -> Int -> T.Tensor -> a -> IO ()
showLog n i i' l p = when (i == 0 || i `mod` n == 0 || i == i' - 1) $ do
    putStrLn $ show i ++ " | " ++ show (T.asValue @Float l) ++ "\n\t" ++ show p

step :: (T.CppOptimizer o) => Int -> (T.Tensor -> T.Tensor) -> DesignVariable
     -> T.CppOptimizerState o -> IO (DesignVariable, T.CppOptimizerState o, Int)
-- step i f p o | i == 0                   = pure (p, o, 0)
--              | T.all (T.lt l (-1.0e-2)) = pure (p, o, i)
--              | otherwise                = do
step 0 _ p o  = pure (p, o, 0)
step i f p o  = do
    -- showLog 1000 i 10000 l p >> T.runStep p o l 1.0e-4 >>= uncurry (step i' f)
    -- showLog 100 i 10000 l p >> T.closureStep p o l >>= uncurry (step i' f)
    -- tic <- getTime Realtime
    -- (p',o') <- T.runStep p o l 1.0e-4 
    -- toc <- getTime Realtime
    -- let rt = (*1.0e-9) . realToFrac . toNanoSecs $ diffTimeSpec toc tic :: Float
    -- print $ "One Step took " ++ show rt ++ "s"
    -- step i' f p' o'
    -- showLog 100 i 2000 l p >> T.runStep p o l 1.0e-3 >>= uncurry (step i' f)
    if T.all (T.lt l (-1.0e-2))
       then pure (p, o, i)
       else T.runStep p o l 1.0e-3 >>= uncurry (step i' f)
  where
    i' = i - 1
    f' (DesignVariable [T.IndependentTensor{..}]) = f toDependent
    f' (DesignVariable _) = error "no parameters"
    l  = f' p

step' :: Int -> (T.Tensor -> T.Tensor) -> DesignVariable -> T.Adam -> IO (DesignVariable, T.Adam, Int)
step' i f p o | i == 0                   = pure (p, o, 0)
              | T.all (T.lt l (-1.0e-2)) = pure (p, o, i)
              | otherwise                = do
    T.runStep p o l 1.0e-4 >>= uncurry (step' i' f)
  where
    i' = i - 1
    f' (DesignVariable [T.IndependentTensor{..}]) = f toDependent
    f' (DesignVariable _) = error "no parameters"
    l  = f' p

run' :: PDK.ID -> CKT.ID -> IO Result
run' pdk ckt = do
    model <- T.unTraceModel <$> T.loadInferenceModel tracePath
    let idx        = T.arange 0 (length pps) 1 $ T.withDType T.Int64 T.defaultOpts
        model'     = T.indexSelect 0 idx . model
        objective' = objective pdk ckt s model'

    var <- T.sample . DesignVarSpec $ length dps
    -- opt <- T.initOptimizer adamOpt $ T.flattenParameters var
    opt <- T.initOptimizer adamOpt var
    -- opt <- T.initOptimizer lbfgsOpt var
    -- let opt = T.mkAdam 0 0.9 0.999 $ T.flattenParameters var

    -- (rt', (DesignVariable obj', opt', iter')) <- timeItT $ step numIter objective' var opt
    -- let rt = (*1.0e-3) . double2Float $ rt'
    tic <- getTime Realtime
    (DesignVariable !obj', _, !iter') <- step numIter objective' var opt
    toc <- getTime Realtime
    let !rt = (*1.0e-9) . realToFrac . toNanoSecs $ diffTimeSpec toc tic :: Float
    putStrLn $ "Took " ++ show rt ++ "s"

    x <- trafoX pdk ckt . head <$> mapM ((T.clone <=< T.detach) . T.toDependent) obj'
    let y = model x
        l = T.asValue . objective' . T.toDependent $ head obj'
        --z = T.sliceDim 0 (len + 2) (len + len' + 2) 1 y
        z = T.sliceDim 0 len (len + len') 1 y
        dfX = DF.union (DF.DataFrame dps $ T.reshape [1,-1] x)
                       (DF.DataFrame vds $ T.reshape [1,-1] z)
        dfY = DF.DataFrame pps . T.reshape [1,-1] $ T.sliceDim 0 0 len 1 y
        df  = DF.union dfX dfY

    df' <- validateResult pdk ckt df

    pure $ Result df df' l rt (numIter - iter')
  where
    (vds, dps) = partition (isSuffixOf "_vds") $ CKT.electricalParams ckt
    pps        = [ "a_0", "ugbw", "pm", "gm", "sr_r", "psrr_p", "psrr_n"
                 , "cmrr", "sr_f", "voff_stat", "idd" ]
    -- tracePath  = "./models/" ++ show ckt ++ "-" ++ show pdk ++ "-trace-" ++ activ ++ ".pt"
    tracePath  = "./models/" ++ show ckt ++ "-" ++ show pdk ++ "-stat-trace.pt"
    s'         = specification pdk ckt
    len        = length s'
    len'       = length vds
    s          = T.asTensor @[Float] s'
    numIter    = 2000
    adamOpt    = def { T.adamLr          = 1.0e-3
                     , T.adamBetas       = (0.9, 0.999)
                     , T.adamEps         = 1.0e-8
                     , T.adamWeightDecay = 0.0
                     , T.adamAmsgrad     = True
                     } :: T.AdamOptions
    -- lbfgsOpt   = def { T.lbfgsLr = 1
    --                  , T.lbfgsMaxIter = 20
    --                  , T.lbfgsMaxEval = (20 * 5) `div` 4
    --                  , T.lbfgsToleranceGrad = 1.0e-7
    --                  , T.lbfgsToleranceChange = 1.0e-9
    --                  , T.lbfgsHistorySize = 100
    --                  , T.lbfgsLineSearchFn = Nothing }

run :: Int -> PDK.ID -> CKT.ID -> IO ()
run num pdk ckt = replicateM num (run' pdk ckt) >>= evalResults pdk ckt "adam"
