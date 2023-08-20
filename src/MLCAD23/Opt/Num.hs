{-# OPTIONS_GHC -Wall -fplugin Debug.Breakpoint #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Multi-Objective Optimization
module MLCAD23.Opt.Num where

import qualified Numeric.LinearAlgebra      as A
import           Numeric.GSL.Minimization
import           Control.Monad                    (replicateM)
import           GHC.Float
import           Data.List                        (isSuffixOf, partition)
import           System.Clock
import           System.IO.Unsafe
import qualified Data.Frame                 as DF
import qualified Torch                      as T
import qualified Torch.Extensions           as T

-- import           Debug.Breakpoint
import qualified MLCAD23.Config.CKT         as CKT
import qualified MLCAD23.Config.PDK         as PDK

import           MLCAD23.Opt

objective' :: PDK.ID -> CKT.ID -> T.Tensor -> (T.Tensor -> T.Tensor)
          -> ([Double] -> [Double])
objective' pdk ckt spec fun = obj'
  where
    obj' :: [Double] -> [Double]
    obj' xs = T.asValue . T.squeezeAll . T.cat (T.Dim 0) $ T.grad y' [xs']
      where 
        xs' = unsafeDupablePerformIO . T.makeIndependent $ T.asTensor xs
        y'  = T.sumAll . losses spec . fun . trafoX pdk ckt
            . T.clamp 0.0 1.0 . T.toDType T.Float $ T.toDependent xs'

objective :: PDK.ID -> CKT.ID -> T.Tensor -> (T.Tensor -> T.Tensor)
          -> ([Double] -> Double)
objective pdk ckt spec fun = obj
  where
    obj :: [Double] -> Double
    obj = T.asValue . T.toDType T.Double . losses spec . fun
        . trafoX pdk ckt . T.clamp 0.0 1.0 . T.toDType T.Float . T.asTensor
    
run' :: Optimizer -> PDK.ID -> CKT.ID -> IO Result
run' opt pdk ckt = do
    model <- T.unTraceModel <$> T.loadInferenceModel tracePath
    let idx    = T.arange 0 (length pps) 1 $ T.withDType T.Int64 T.defaultOpts
        model' = T.indexSelect 0 idx . model
        f      = objective pdk ckt s' model'
        f'     = objective' pdk ckt s' model'

    start <- T.asValue . T.toDType T.Double <$> T.normalWithSizeIO 0.5 0.1 (length dps)
    tic <- getTime Realtime
    let (!sol, !pth) = if opt == NumericD then minimizeD VectorBFGS2 precision numIter step tol f f' start
                                          else minimize  NMSimplex2  precision numIter sizes f start
        !iter' = fst $ A.size pth
    -- putStr $ show iter' ++ " steps took "
    toc <- getTime Realtime
    let !rt = (*1.0e-9) . realToFrac . toNanoSecs $ diffTimeSpec toc tic :: Float

    putStrLn $ "Took " ++ show rt ++ "s"

    let x = trafoX pdk ckt . T.clamp 0.0 1.0 . T.toDType T.Float $ T.asTensor sol
        y = model x
        l = double2Float $ f sol
        z = T.sliceDim 0 len (len + len') 1 y
        dfX = DF.union (DF.DataFrame dps $ T.reshape [1,-1] x)
                       (DF.DataFrame vds $ T.reshape [1,-1] z)
        dfY = DF.DataFrame pps . T.reshape [1,-1] $ T.sliceDim 0 0 len 1 y
        df  = DF.union dfX dfY

    !df' <- validateResult pdk ckt df
    let !res' = Result df df' l rt iter'

    pure res' --  $ Result df df' l rt iter'
  where
    (vds, dps) = partition (isSuffixOf "_vds") $ CKT.electricalParams ckt
    pps        = [ "a_0", "ugbw", "pm", "gm", "sr_r", "psrr_p", "psrr_n"
                 , "cmrr", "sr_f", "voff_stat", "idd" ]
    tracePath  = "./models/" ++ show ckt ++ "-" ++ show pdk ++ "-stat-trace.pt"
    s'         = T.asTensor $ specification pdk ckt
    len        = head $ T.shape s'
    len'       = length vds
    numIter    = 2000
    precision  = 1.0e-3
    sizes      = replicate @Double (length dps) 0.2
    step       = 0.001
    tol        = 0.1

run :: Int -> Optimizer -> PDK.ID -> CKT.ID -> IO ()
run num opt pdk ckt = do
    replicateM num (run' opt pdk ckt) >>= evalResults pdk ckt ext
  where
    ext = if opt == NumericD then "lbfgs" else "nm"

