{-# OPTIONS_GHC -Wall -fplugin Debug.Breakpoint #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Multi-Objective Optimization
module MLCAD23.Opt where

import           Data.List                       (isSuffixOf)
import           Data.Frame                      ((??))
import qualified Data.Frame                as DF
import qualified Torch                     as T
import qualified Torch.Extensions          as T

import qualified MLCAD23.Config.CKT        as CKT
import qualified MLCAD23.Config.PDK        as PDK
import qualified MLCAD23.Elec              as E

-- import           Debug.Breakpoint

-- | Available types of Optimization
data Optimizer = Numeric    -- ^ Numeric w/o derivative
               | NumericD   -- ^ Numeric with derivative
               | Gradient   -- ^ Gradient Descent with AD
               | GeneticS   -- ^ Genetic Single Objective
    deriving (Show, Eq, Read)

-- | Optimization Result
data Result = Result { mdlData   :: !(DF.DataFrame T.Tensor) -- ^ Candidate solution prediction
                     , simData   :: !(DF.DataFrame T.Tensor) -- ^ Candidate solution simulated
                     , finalLoss :: !Float                   -- ^ Final Loss Value
                     , runTime   :: !Float                   -- ^ Run Time
                     , fEvals    :: !Int                     -- ^ Number of Function Evaluations
                     } deriving (Show)

--  "a_0", "ugbw", "pm", "gm", "sr_r", "psrr_p", "psrr_n", "cmrr", "sr_f", "voff_stat", "idd"
specification :: PDK.ID -> CKT.ID -> [Float]
specification _ CKT.SYM = [55.0, 3.5e6, 60.0, -25.0, 3.5e6, 75.0, 60.0, 80.0, -3.5e6, 3.0e-3, 5.0e-5]
specification _ CKT.MIL = [100.0, 4.0e6, 70.0, -45.0, 2.5e6, 100.0, 100.0, 100.0, -2.5e6, 3.0e-3, 1.0e-4]
specification _ CKT.RFA = [75.0, 2.5e6, 75.0, -50.0, 1.5e6, 100.0, 85.0, 100.0, -1.5e-6, 1.5e-3, 5.0e-5 ]

-- | Lower bounds
xMins :: PDK.ID -> CKT.ID -> T.Tensor
xMins _ CKT.SYM = T.asTensor @[Float] [ 6.0, 6.0, 6.0, 6.0 -- gm/Id
                                      , 6.5, 6.5, 6.5, 6.5 ]-- log10 fug
xMins _ CKT.MIL = T.asTensor @[Float] [ 6.0, 6.0, 6.0, 6.0 -- gm/Id
                                      , 6.5, 6.5, 6.5, 6.5 ]-- log10 fug
xMins _ CKT.RFA = T.asTensor @[Float] [ 7.0, 7.0, 5.0, 5.0, 6.0, 6.0, 7.0 -- gm/Id
                                      , 6.0, 6.0, 6.0, 6.5, 6.0, 6.5, 6.5 ]-- log10 fug

-- | Upper bounds
xMaxs :: PDK.ID -> CKT.ID -> T.Tensor
xMaxs _ CKT.SYM = T.asTensor @[Float] [ 18.0, 18.0, 18.0, 18.0 -- gm/Id
                                      , 9.0, 9.0, 9.0, 9.0 ]   -- log10 fug
xMaxs _ CKT.MIL = T.asTensor @[Float] [ 18.0, 18.0, 18.0, 18.0 -- gm/Id
                                      , 9.0, 9.0, 9.0, 9.0 ]   -- log10 fug
xMaxs _ CKT.RFA = T.asTensor @[Float] [ 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0 -- gm/Id
                                      , 10.0, 9.5, 10.0, 9.5, 9.5, 9.6, 9.5 ]   -- log10 fug

xMask :: CKT.ID -> T.Tensor
xMask CKT.SYM = T.asTensor @[Bool] (replicate 4 False ++ replicate 4 True)
xMask CKT.MIL = T.asTensor @[Bool] (replicate 4 False ++ replicate 4 True)
xMask CKT.RFA = T.asTensor @[Bool] (replicate 7 False ++ replicate 7 True)

-- | Lower bounds
yMins :: PDK.ID -> CKT.ID -> T.Tensor
yMins _ CKT.SYM = T.asTensor @[Float] [20.0, 5.5, 0.0, -100.0, 5.5, 50.0, 20.0, 45.0, 5.5, 0.0, 0.0]
yMins _ CKT.MIL = T.asTensor @[Float] [50.0, 5.2, 0.0, -150.0, 5.5, 60.0, 40.0, 50.0, 5.2, 0.0, 0.0]
yMins _ CKT.RFA = T.asTensor @[Float] [25.0, 5.2, 30.0, -140.0, 4.5, 40.0, 20.0, 30.0, 4.2, 0.0, 0.0]

-- | Upper bounds
yMaxs :: PDK.ID -> CKT.ID -> T.Tensor
yMaxs _ CKT.SYM = T.asTensor @[Float] [75.0, 7.2, 100.0, 0.0, 7.2, 100.0, 75.0, 150.0, 7.2, 55.0e-3, 60.0e-6]
yMaxs _ CKT.MIL = T.asTensor @[Float] [150.0, 7.0, 100.0, 0.0, 8.0, 150.0, 180.0, 140.0, 6.8, 50.0e-3, 250.0e-6]
yMaxs _ CKT.RFA = T.asTensor @[Float] [85.0, 6.8, 100.0, 0.0, 6.5, 170.0, 130.0, 170.0, 6.6, 0.1, 50.0e-6]

yMask :: PDK.ID -> CKT.ID -> T.Tensor
yMask _ _ = T.asTensor @[Bool] [ False, True, False, False, True, False
                               , False, False, True, False, False ]

iBranch :: PDK.ID -> CKT.ID -> [T.Tensor]
iBranch _ CKT.SYM = [ T.asTensor @[Float] [i * 3.0e-6  | i <- [1 .. 3]]
                    , T.asTensor @[Float] [i * 3.0e-6  | i <- [3 .. 8]] ]
iBranch _ CKT.MIL = [ T.asTensor @[Float] [i * 3.0e-6  | i <- [1 .. 4]] 
                    , T.asTensor @[Float] [i * 30.0e-6 | i <- [1 .. 4]] ]
iBranch _ CKT.RFA = [ T.asTensor @[Float] [i * 3.0e-6  | i <- [1 .. 3]] 
                    , T.asTensor @[Float] [i * 3.0e-6  | i <- [2 .. 4]] ]

-- | Inputs from [0,1] to Real
trafoX :: PDK.ID -> CKT.ID -> T.Tensor -> T.Tensor
trafoX pdk ckt x = T.cat (T.Dim 0) [ op, ib ]
  where
    lb   = xMins pdk ckt
    ub   = xMaxs pdk ckt
    msk  = xMask ckt
    ibs  = iBranch pdk ckt
    idx  = head $ T.shape msk
    op'  = T.indexSelect' 0 [0 .. pred idx] x
    ib'  = T.split 1 (T.Dim 0) $ T.indexSelect' 0 [idx .. pred idx + length ibs] x
    op   = T.trafo' msk $ T.scale' lb ub op'
    ib'' = T.cat (T.Dim 0) $ zipWith T.δround ibs ib'
    ib   = if ckt == CKT.RFA
              then ib'' * T.asTensor @[Float] [0.916, 1.0]
              else ib''

-- | Inputs from [0,1] to Real
trafoX' :: PDK.ID -> CKT.ID -> [T.Tensor] -> T.Tensor
trafoX' pdk ckt xs = T.cat (T.Dim 0) [ op, ib ]
  where
    lb  = xMins pdk ckt
    ub  = xMaxs pdk ckt
    msk = xMask ckt
    op  = T.trafo' msk . T.scale' lb ub $ head xs
    ib  = T.cat (T.Dim 0) $ zipWith T.δround (iBranch pdk ckt) (drop 1 xs)

-- | Outputs from real to [0,1]
trafoY :: PDK.ID -> CKT.ID -> (T.Tensor -> T.Tensor)
trafoY pdk ckt = T.scale lb ub . T.trafo msk
  where
    msk = yMask pdk ckt
    lb  = yMins pdk ckt
    ub  = yMaxs pdk ckt

-- | Outputs from [0,1] to real
trafoY' :: PDK.ID -> CKT.ID -> [Float] -> T.Tensor
trafoY' pdk ckt = T.scale (yMins pdk ckt) (yMaxs pdk ckt) . T.asTensor

-- | Performance Predicate
predicate :: T.Tensor
predicate = T.asTensor @[Bool] [ True, True, True, False, True, True
                               , True, True, False, False, False ]

losses' :: T.Tensor -> T.Tensor -> T.Tensor
losses' x' x = l'
  where
    m   = T.toDType T.Float predicate
    m'  = 1.0 - m 
    l'  = ((m * (x' - x)) + (m' * (x - x'))) / T.abs x'

losses :: T.Tensor -> T.Tensor -> T.Tensor
losses x' x = if T.all (T.le c 0.0) then c' else c
  where
    l'  = losses' x' x
    c   = T.sumAll $ T.relu l'
    c'  = T.sumAll $ T.leakyRelu 0.01 l'

validateResult :: PDK.ID -> CKT.ID -> DF.DataFrame T.Tensor
               -> IO (DF.DataFrame T.Tensor)
validateResult pdk ckt df = do
    op <- E.mkOpAmp 1 ckt pdk PDK.MC
    let elec = DF.lookup (CKT.electricalParams ckt) df
        tb   = DF.DataFrame ["vdd", "i0", "temp", "cl"]
             $ T.asTensor @[[Float]] [[2.5, 3.0e-6, 27.0, 5.0e-12]]
    E.simulate' op $ DF.union elec tb

-- | Load capancitance depending on circuit
cl :: CKT.ID -> Float
cl CKT.MIL = 15.0e-12
cl _       =  5.0e-12

defaultPVT :: T.Tensor -> T.Tensor
defaultPVT x = T.cat (T.Dim 0) [x, d]
  where
    d  = T.asTensor @[Float] [2.5, 3.0e-6, 5.0e-6, 27.0]

evalResults :: PDK.ID -> CKT.ID -> String -> [Result] -> IO ()
evalResults pdk ckt opt res = do
    let df    = DF.concat $ map mdlData res
        df'   = DF.concat $ map simData res
        ts    = map runTime res
        ts'   = T.reshape [-1,1] $ T.asTensor ts
        is    = map fEvals res
        is'   = T.toDType T.Float . T.reshape [-1,1] $ T.asTensor is
        ls    = map finalLoss res
        ls'   = T.reshape [-1,1] $ T.asTensor ls

    let y     = DF.values . DF.lookup pps $ df
        y'    = DF.values . DF.lookup pps $ df'
        -- z     = T.squeezeAll . DF.values . DF.lookup vds $ df
        x     = T.squeezeAll . DF.values . DF.lookup dps $ df
        x'    = T.squeezeAll . DF.values . DF.lookup dps $ df'
        -- z'    = T.squeezeAll . DF.values . DF.lookup vds $ df'
        l     = losses' s y
        l'    = losses' s y'
        fp    = T.toDType T.Float $ T.logicalAnd (T.le l 0.0) (T.gt l' 0.0)
        fp'   = T.sumDim (T.Dim 1) T.KeepDim T.Float fp
        -- fp'   = T.reshape [-1,1] $ T.sumAll fp
        err'  = T.meanDim (T.Dim 1) T.KeepDim T.Float $ T.abs $ (y - y') / y'
        -- err'  = T.reshape [-1,1] . T.meanAll . T.abs $ (y - y') / y'
        fom   = flip T.div (T.mulScalar @Float 1.0e3 (df ?? "idd"))
              . T.mulScalar (1.0e12 * cl ckt) $ T.mulScalar @Float 1.0e-6 (df ?? "ugbw")
        fom'  = T.reshape [-1,1] fom

    let sim = DF.insert ["fom", "false", "err"] (T.cat (T.Dim 1) [fom', fp', err']) df'
        mdl = DF.insert ["time", "calls", "loss"] (T.cat (T.Dim 1) [ts', is', ls']) df

    DF.toCSV ("/mnt/data/share/ml-exchange/iccad/" ++ show ckt ++ "_sizings_" ++ opt ++ "_header.csv") mdl
    DF.toCSV ("/mnt/data/share/ml-exchange/iccad/" ++ show ckt ++ "_sizings_simulated_" ++ opt ++ "_header.csv") sim

    let top = -- DF.rowFilter (T.squeezeAll $ T.lt (sim ?? "gm") 0.0)
            DF.union (DF.lookup ["fom", "area", "voff_stat", "false", "err"] sim)
                       (DF.lookup ["time", "calls"] mdl)

    let avgFM = T.asValue @Float . T.meanAll . T.squeezeAll $ top ?? "fom"
        avgAE = T.asValue @Float . T.meanAll . T.squeezeAll $ top ?? "area"
        avgVO = T.asValue @Float . T.meanAll . T.squeezeAll $ top ?? "voff_stat"
        avgRT = T.asValue @Float . T.meanAll . T.squeezeAll $ top ?? "time"
        avgST = T.asValue @Float . T.meanAll . T.squeezeAll $ top ?? "calls"
        avgFP = T.asValue @Float . T.meanAll . T.squeezeAll $ top ?? "false"
        meanE = T.asValue @Float . T.meanAll . T.squeezeAll . (*100) $ top ?? "err"
        -- meanE = T.asValue @Float . T.meanAll . T.mulScalar @Float 100 . T.abs . T.mul (T.gt l' 0.0) $ (y - y') / y'
        -- meanE = T.asValue @Float . T.meanAllNZ . T.mulScalar @Float 100 . T.abs . T.mul fp $ (y - y') / y'

    putStrLn ""
    print x
    putStrLn ""
    print x'
    putStrLn ""
    print s
    putStrLn ""
    print y
    putStrLn ""
    print y'
    putStrLn ""
    print . T.mulScalar @Float 100 $ T.abs $ (y - y') / y'
    print . T.meanDim (T.Dim 0) T.RemoveDim T.Float . T.mulScalar @Float 100 $ T.abs $ (y - y') / y'

    putStrLn ""
    putStrLn $ show pdk ++ " " ++ show ckt ++ ":"
    putStrLn "opt,fom,area,voff,time,calls,false,err"
    putStrLn $ opt ++ "," ++ show avgFM ++ "," ++ show avgAE ++ ","
                   ++ show avgVO ++ "," ++ show avgRT ++ "," ++ show avgST
                   ++ "," ++ show avgFP ++ "," ++ show meanE
    pure ()
  where
    -- gps        = CKT.geometricalParams ckt
    -- (vds, dps) = partition (isSuffixOf "_vds") $ CKT.electricalParams ckt
    dps        = filter (not . isSuffixOf "_vds") $ CKT.electricalParams ckt
    pps        = [ "a_0", "ugbw", "pm", "gm", "sr_r", "psrr_p", "psrr_n"
                 , "cmrr", "sr_f", "voff_stat", "idd" ]
    s'         = specification pdk ckt
    -- len        = length s'
    s          = T.asTensor @[Float] s'
