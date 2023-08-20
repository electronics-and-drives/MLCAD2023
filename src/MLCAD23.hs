{-# OPTIONS_GHC -Wall -fplugin Debug.Breakpoint #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE StrictData #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DuplicateRecordFields #-}

-- | Functional Analog Circuit Design
module MLCAD23 where

-- import           Control.Monad
import           Data.List
import qualified Data.Map                    as M
import qualified Data.Vector.Storable        as V
import qualified Torch                       as T
-- import qualified Torch.Extensions            as T
import           Data.List.Split                      (chunksOf)
import           Data.Frame                           ((??))
import qualified Data.Frame                  as DF
import qualified MLCAD23.Elec                as E
import qualified MLCAD23.Geom                as G
import qualified MLCAD23.Untyped.Train       as Train
import qualified MLCAD23.Viz                 as Viz
import qualified MLCAD23.Opt                 as Opt
import qualified MLCAD23.Opt.Num             as NU
import qualified MLCAD23.Opt.Soo             as SO
import qualified MLCAD23.Opt.Grad            as GR
import qualified MLCAD23.Config.CKT          as CKT
import qualified MLCAD23.Config.PDK          as PDK

-- import           Debug.Breakpoint

step :: E.OpAmp -> [DF.DataFrame T.Tensor] -> [DF.DataFrame T.Tensor]
     -> IO (DF.DataFrame T.Tensor)
step  _    []   !ys = pure $! DF.concat ys
step !ops (x:xs) !ys = do
    putStr $ show (length xs) ++ " "
    !y <- DF.rowSelect' [0 .. n - 1] <$> E.simulate' ops x 
    let !ys' = y:ys
    step ops xs ys'
  where
    n = DF.nRows x

collectData' :: Int -> CKT.ID -> PDK.ID -> PDK.Corner -> Int
             -> IO (DF.DataFrame T.Tensor)
collectData' par ckt pdk corner size = do
    !ops <- E.mkOpAmp par' ckt pdk corner
    putStrLn $ "Starting " ++ show size ++ " chunk for " ++ show ckt ++ " in "
                           ++ show pdk ++ " (" ++ show corner ++ ") with "
                           ++ show (length (E.opAmps ops)) ++ " parallel spectre sessions."

    !inputs  <- E.randomBehaviour ops size pvt
    !sizings <- E.simulate ops inputs
    !outputs <- E.results ops sizings

    E.close ops

    -- pure outputs
    let corner' = T.reshape [-1,1] . T.toDType T.Float . T.asTensor
                . replicate (DF.nRows outputs) $ fromEnum corner

    pure $ DF.insert ["corner"] corner' outputs

  where
    par' = if par > size then size else par
    pvt  = True

collectData :: Int -> CKT.ID -> PDK.ID -> PDK.Corner -> Int -> IO ()
collectData num ckt pdk corner size = do
    !outputs <- mapM (collectData' num ckt pdk corner)
                $ replicate (div size sizeMax) sizeMax ++ [rest | rest > 0]
    DF.toFile path $ DF.concat outputs
  where
    sizeMax = 10000
    rest    = size `rem` sizeMax
    path    = "./data/" ++ show ckt ++ "-" ++ show pdk ++ "-" ++ show corner

collectCornerData' :: Int -> CKT.ID -> PDK.ID -> Int -> DF.DataFrame T.Tensor
                   -> PDK.Corner -> IO (DF.DataFrame T.Tensor)
collectCornerData' par ckt pdk size sizing corner = do
    putStrLn $ "Starting " ++ show size ++ " chunk for " ++ show ckt ++ " in "
                           ++ show pdk ++ " (" ++ show corner ++ ")"
    ops <- E.mkOpAmp par' ckt pdk corner
    mapM_ (G.simulate (E.opAmps ops)) geom

    prf <- DF.rowTake (DF.nRows sizing) . DF.concat . map DF.fromParameters
            <$> G.readResults (E.opAmps ops) geomT

    let corner' = T.reshape [-1,1] . T.toDType T.Float . T.asTensor
                . replicate (DF.nRows prf) $ fromEnum corner
    E.close ops
    pure . DF.insert ["corner"] corner' $ DF.union sizing prf
  where
    par' = if par > size then size else par
    geom = map DF.asParameters . DF.chunksOf par' $ sizing
    geomT = ( map (M.map V.fromList . M.fromList
            . zip (M.keys (head geom)) . chunksOf (length geom))
            . transpose . foldl1 (++) . transpose)
                (map (M.elems . M.map V.toList) geom)

collectCornerData :: Int -> CKT.ID -> PDK.ID -> Int -> IO ()
collectCornerData num ckt pdk size = do
    !dfTM <- DF.concat <$> mapM (collectData' num ckt pdk PDK.TM) chunks
    let df'  = DF.lookup params dfTM
        area = T.reshape [-1,1] $ dfTM ?? "area"

    dfCorners' <- mapM (collectCornerData' num ckt pdk size df' . toEnum) [ 1 .. 4 ]
                        -- [PDK.WZ, PDK.WO, PDK.WP, PDK.WS]

    let dfCorners = DF.lookup (DF.columns dfTM) . DF.concat
                  $ map (DF.insert ["area"] area) dfCorners'

    DF.toFile path $ DF.concat [dfTM, dfCorners]
    pure ()
  where
    sizeMax = 10000
    rest    = size `rem` sizeMax
    chunks  = replicate (div size sizeMax) sizeMax ++ [rest | rest > 0]
    path    = "./data/" ++ show ckt ++ "-" ++ show pdk ++ "-corners"
    params  = CKT.geometricalParams ckt ++ ["vdd", "i0", "cl", "temp"]

trainModel :: Int -> CKT.ID -> PDK.ID -> PDK.Corner -> IO ()
trainModel = Train.train 
-- trainModel = Train'.train 

optimCircuit :: Opt.Optimizer -> Int -> CKT.ID -> PDK.ID -> IO ()
optimCircuit Opt.Gradient     runs ckt pdk = GR.run runs pdk ckt
optimCircuit opt@Opt.Numeric  runs ckt pdk = NU.run runs opt pdk ckt
optimCircuit opt@Opt.NumericD runs ckt pdk = NU.run runs opt pdk ckt
optimCircuit Opt.GeneticS     runs ckt pdk = SO.run runs pdk ckt

visualizeData :: PDK.ID -> CKT.ID -> IO ()
visualizeData = Viz.visualizeModel

compareCircuits :: PDK.ID -> IO ()
compareCircuits pdk = Viz.compareCircuits pdk [CKT.SYM, CKT.MIL, CKT.RFA]

run' :: Mode -> CKT.ID -> PDK.ID -> PDK.Corner -> Opt.Optimizer -> Int -> Int -> IO ()
run' Sample   ckt pdk crn _   par num = collectData par ckt pdk crn num
run' Train    ckt pdk crn _   _   num = trainModel num ckt pdk crn
run' Optimize ckt pdk _   opt _   num = optimCircuit opt num ckt pdk 

run :: Args -> IO ()
run Args{..} = do
    run' mode' ckt pdk crn opt numParallel num
    pure ()
  where
    mode' = read @Mode mode
    opt   = read @Opt.Optimizer optim
    pdk   = read @PDK.ID pdkID
    ckt   = read @CKT.ID cktID
    crn   = read @PDK.Corner corner
    num   = case mode' of
                 Sample   -> numPoints
                 Train    -> numEpochs
                 Optimize -> numRuns

-- | Mode of operation
data Mode = Sample      -- ^ Sample data
          | Train       -- ^ Train a surrogate model
          | Optimize    -- ^ Optimize circuit using surrogate model
          deriving (Show, Read, Eq)

-- | Commandline Arguments
data Args = Args { mode        :: !String -- ^ What to do
                 , numParallel :: !Int    -- ^ Number of parallel spectre sessions
                 , numPoints   :: !Int    -- ^ Number of data points
                 , numRuns     :: !Int    -- ^ Number of Runs
                 , numEpochs   :: !Int    -- ^ Number of training epochs
                 , optim       :: !String -- ^ Optimizer
                 , cktID       :: !String -- ^ Circuit ID (SYM,MIL,RFA)
                 , pdkID       :: !String -- ^ PDK ID (GPDK045, GPDK090, XT018, XH018)
                 , corner      :: !String -- ^ Process corner
                 } deriving (Show)
