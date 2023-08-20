{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE StrictData #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DuplicateRecordFields #-}

-- | Functional Analog Circuit Design
module MLCAD23.Geom where

import           GHC.Float                          (double2Float)
-- import           Control.Concurrent.Async
import           Control.DeepSeq
import           Control.Monad
import           Control.Parallel.Strategies
import           Control.Scheduler                  ( Comp (..)
                                                    , traverseConcurrently
                                                    , traverseConcurrently_
                                                    , replicateConcurrently )
import           Data.List                          (partition)
import           Data.List.Split                    (chunksOf)
import           Data.Map                           ((!))
import qualified Data.Map                    as M
import           Data.Maybe
import qualified Data.NutMeg                 as N
import qualified Data.Vector.Storable        as V
import           Spectre.Interactive
-- import           Spectre                            (Analysis (..))
import           System.Random

import qualified Torch                       as T
-- import qualified Torch.Extensions            as T
import           Data.Frame                         ((??))
import qualified Data.Frame                  as DF

import qualified MLCAD23.Config.PDK          as PDK
import qualified MLCAD23.Config.CKT          as CKT
import           MLCAD23.Internal
import           MLCAD23.Util

-- | Single Ended Operational Amplifier
data OpAmp = OpAmp { session      :: !Session 
                   , simDir       :: !FilePath
                   , parameters   :: !(M.Map String Double)
                   , geomInit     :: !(M.Map String Double)
                   , areaExpr     :: !String
                   , constraints  :: !PDK.Constraints
                   , devices      :: !(M.Map String CKT.DeviceType)
                   , dcopParams   :: !(M.Map String String)
                   , offsParams   :: !(M.Map String String)
                   , performances :: !(M.Map String String) }

-- | Alias for List of OpAmps
type OpAmps = [OpAmp]

-- | Perform pure computation in parallel
parallel :: (OpAmp -> Parameters) -> OpAmps -> Parameters
parallel func amps = M.unionsWith (V.++) (map func amps `using` parList rdeepseq)

-- | Perform IO action in parallel
parallelIO :: (OpAmp -> IO Parameters) -> OpAmps -> IO Parameters
parallelIO func amps = M.unionsWith (V.++) <$> traverseConcurrently (ParN n) func amps
  where
    n = fromIntegral $ length amps

-- | Construct an OpAmp
mkOpAmp :: CKT.ID -> PDK.ID -> IO OpAmp
mkOpAmp ckt pdk = head <$> mkOpAmps 1 ckt pdk corner
  where
    corner = PDK.TM

-- | Construct a number of same OpAmps
mkOpAmps :: Int -> CKT.ID -> PDK.ID -> PDK.Corner -> IO OpAmps
mkOpAmps num cktId pdkId corner = replicateConcurrently (ParN num') num 
                                $ mkOpAmp' cktId pdkId corner ckt pdk net
  where
    num' = fromIntegral num
    pdk  = circusHome ++ "/pdk/" ++ show pdkId ++ ".yml"
    ckt  = circusHome ++ "/ckt/" ++ show cktId ++ ".yml"
    net  = circusHome ++ "/pdk/" ++ show pdkId ++ "/" ++ show cktId ++ ".scs"

updateCorner :: PDK.ID -> PDK.Corner -> PDK.CFG -> PDK.CFG
updateCorner  pdk corner cfg@PDK.CFG{..} = cfg {PDK.include = incs}
  where
    incs = [ if PDK.section i == "mc_g"
                then i {PDK.section = PDK.showCorner pdk corner} else i
           | i <- include ]
    
-- | Actual Constructor for OpAmps
mkOpAmp' :: CKT.ID -> PDK.ID -> PDK.Corner -> FilePath -> FilePath -> FilePath
         -> IO OpAmp
mkOpAmp' cktId pdkId corner cktPath pdkPath net = do
    ckt     <- CKT.readConfig cktPath 
    pdk     <- updateCorner pdkId corner <$> PDK.readConfig pdkPath
    simDir' <- setupDir cktId pdkId ckt pdk net

    let netPath      = simDir' ++ "/tb.scs"
        parameters'  = M.unions [ CKT.testbench $ CKT.parameters ckt 
                                , PDK.testbench pdk, PDK.defaults pdk ]
        geomInit'    = CKT.geometrical . CKT.parameters $ ckt
        areaExpr'    = CKT.area        . CKT.parameters $ ckt
        constraints' = PDK.constraints pdk
        devices'     = M.fromList . map asTuple . CKT.devices $ ckt
        dcOpps       = PDK.parameters . PDK.dcop . PDK.devices $ pdk
        opPre        = PDK.prefix (PDK.dcop . PDK.devices $ pdk :: PDK.DCOPConfig)
        opSuf        = PDK.suffix (PDK.dcop . PDK.devices $ pdk :: PDK.DCOPConfig)
        dcopParams'  =  M.fromList [ (opPre ++ d ++ opSuf ++ ":" ++ op, d ++ "_" ++ op)
                                   | op <- dcOpps
                                   , d  <-  M.keys . M.filter isMOS $ devices' ]
        ofDevs       = PDK.dcmatch . PDK.devices $ pdk
        offsParams'  = M.unions [ M.fromList [ ( PDK.prefix (of' :: PDK.DCMConfig) 
                                                    ++ fst dt ++ 
                                                 PDK.suffix (of' :: PDK.DCMConfig)
                                               , fst dt ++ "_" ++ PDK.reference of' )
                                             | of' <- dcmConfig ofDevs . snd $ dt ]
                                | dt <- M.toList . M.filter isMOS $ devices' ]

    sess   <- startSession [] netPath simDir'

    pure OpAmp { session      = sess
               , simDir       = simDir'
               , parameters   = parameters'
               , geomInit     = geomInit'
               , areaExpr     = areaExpr'
               , constraints  = constraints'
               , devices      = devices'
               , dcopParams   = dcopParams'
               , offsParams   = offsParams'
               , performances = performanceParameters }
  where
    isMOS :: CKT.DeviceType -> Bool
    isMOS CKT.NMOS = True
    isMOS CKT.PMOS = True
    isMOS _        = False

    asTuple :: CKT.Device -> (String, CKT.DeviceType)
    asTuple CKT.Device{..} = (id', type')

-- | Initial sizing for a single OpAmp
initialSizing' :: OpAmp -> Parameters
initialSizing' OpAmp{..} = M.map V.singleton geomInit

-- | Initial sizing
initialSizing :: OpAmps -> Parameters
initialSizing = parallel initialSizing'

-- | Obtain random sizing parameters
randomSizing :: OpAmps -> IO Parameters
randomSizing = parallelIO randomSizing'

-- | Obtain random sizing parameters for a single OpAmp
randomSizing' :: OpAmp -> IO Parameters
randomSizing' OpAmp{..} = do
    values <- sequence [ V.fromList <$> rng k | k <- keys ]
    pure . M.fromList $ zip keys values
  where
    keys = M.keys geomInit
    minL = PDK.min' . PDK.length $ constraints
    maxL = PDK.max' . PDK.length $ constraints
    grdL = PDK.grid . PDK.length $ constraints
    allL = [minL, (minL + grdL) .. maxL]
    numL = length allL
    minW = PDK.min' . PDK.width  $ constraints
    maxW = PDK.max' . PDK.width  $ constraints
    grdW = PDK.grid . PDK.width  $ constraints
    allW = [minW, (minW + grdW) .. maxW]
    numW = length allW
    -- allM = [1,2 .. 20]
    allM = [1 .. 4]
    numM = length allM
    rng :: String -> IO [Double]
    rng (i:_) | i == 'L' = (:[]) . (allL!!) <$> randomRIO (0, numL - 1) 
              | i == 'W' = (:[]) . (allW!!) <$> randomRIO (0, numW - 1) 
              | i == 'M' = (:[]) . (allM!!) <$> randomRIO (0, numM - 1) 
    rng _                = undefined

-- | Current sizing state of the OpAmp
currentSizing' :: OpAmp -> IO Parameters
currentSizing' OpAmp{..} = M.map V.singleton <$> getParameters session keys
  where
    keys = M.keys geomInit

-- | Current sizing state of all OpAmps
currentSizing :: OpAmps -> IO Parameters
currentSizing = parallelIO currentSizing'

-- | Set Parameters for a single OpAmp
setParams' :: OpAmp -> Parameters' -> IO ()
setParams' OpAmp{..} params = void $! setParameters session (M.mapWithKey clp params) 
  where
    !minL = force . PDK.min' . PDK.length $! constraints
    !maxL = force . PDK.max' . PDK.length $! constraints
    !minW = force . PDK.min' . PDK.width  $! constraints
    !maxW = force . PDK.max' . PDK.width  $! constraints
    clp :: String -> Double -> Double
    clp ('L':_) v = max minL . min maxL $! v
    clp ('W':_) v = max minW . min maxW $! v
    clp _       v =                        v

-- | Set Parameters
setParams :: OpAmps -> Parameters -> IO ()
setParams opAmps params = traverseConcurrently_ (ParOn [length opAmps])
                                                (uncurry setParams')
                        . zip opAmps $ transposeParams params
    -- mapConcurrently_ (uncurry setParams') . zip opAmps $ transposeParams params

extractPerformance' :: OpAmp -> Double -> (String, N.Plot) -> Parameters
extractPerformance' _         _   ("dcmatch", plot) = offset $ N.asRealPlot plot
extractPerformance' _         _   ("stb",     plot) = stability $ N.asComplexPlot plot
extractPerformance' OpAmp{..} _   ("tran",    plot) = transient vs $ N.asRealPlot plot
  where
    vs = fromMaybe 0.5 $! M.lookup "vs" parameters
extractPerformance' _         _   ("noise",   plot) = outputReferredNoise $ N.asRealPlot plot
extractPerformance' OpAmp{..} vdd ("dc1",     plot) = outSwingDC vdd dev $ N.asRealPlot plot
  where
    dev = fromMaybe 1.0e-4 $! M.lookup "dev" parameters
extractPerformance' _         _   ("xf",      plot) = rejection $ N.asComplexPlot plot
extractPerformance' OpAmp{..} _   ("dcop",    plot) = operatingPoint dcopParams $ N.asRealPlot plot
extractPerformance' _         _   (analysis,  _)    = error $ "Extraction for analysis "
                                                        ++ analysis ++ " not found"

extractPerformance :: OpAmp -> Parameters' -> N.NutMeg -> Parameters
extractPerformance opAmp params nut = M.unions [perfA, perfB]
  where
    vdd            = params ! "vdd"
    analysis       = ["dc3", "dc4", "ac"]
    (nutsB, nutsA) = partition ((`elem` analysis) . fst) nut
    perfA          = M.unions $ map (extractPerformance' opAmp vdd) nutsA
    a3db           = (`subtract` 3.0) . V.unsafeHead $! perfA ! "a_0"
    dc3            = N.asRealPlot . fromJust $ lookup "dc3" nutsB
    dc4            = N.asRealPlot . fromJust $ lookup "dc4" nutsB
    dc34           = outputCurrent dc3 dc4
    ac             = outSwingAC a3db vdd . N.asComplexPlot . fromJust
                   $ lookup "ac" nutsB
    perfB          = M.unions [dc34, ac]

xp' :: OpAmp -> T.Tensor -> (String, DF.DataFrame T.Tensor) -> DF.DataFrame T.Tensor
xp' _         _   ("dcmatch", plot) = offset' plot
xp' _         _   ("stb",     plot) = stability' plot
xp' OpAmp{..} _   ("tran",    plot) = transient' vs plot
  where
    vs = double2Float . fromMaybe 0.5 $! M.lookup "vs" parameters
xp' _         _   ("noise",   plot) = outputReferredNoise' plot
xp' OpAmp{..} vdd ("dc1",     plot) = outSwingDC' dev vdd plot
  where
    dev = double2Float . fromMaybe 1.0e-4 $! M.lookup "dev" parameters
xp' _         _   ("xf",      plot) = rejection' plot
xp' OpAmp{..} _   ("dcop",    plot) = operatingPoint' dcopParams plot
xp' _         _   (analysis,  _)    = error $ "Extraction for analysis "
                                            ++ analysis ++ " not found"

xp :: OpAmp -> [T.Tensor] -> [(String, DF.DataFrame T.Tensor)] -> [DF.DataFrame T.Tensor]
xp _ [] _ = []
xp _ _ [] = []
xp op (vdd:vdd') nut = perf : xp op vdd' nut'
  where
    analysis         = ["dc3", "dc4", "ac"]
    (plots, nut')    = splitAt 10 nut
    (plotsB, plotsA) = partition ((`elem` analysis) . fst) plots
    perfA            = foldl1 DF.union $ map (xp' op vdd) plotsA
    a3db             = T.subScalar @Float 3.0 $ perfA ?? "a_0"
    dc3              = fromJust $ lookup "dc3" plotsB
    dc4              = fromJust $ lookup "dc4" plotsB
    dc34             = outputCurrent' dc3 dc4
    ac               = outSwingAC' vdd a3db . fromJust $ lookup "ac" plotsB
    perfB            = DF.union dc34 ac
    perf             = DF.union perfA perfB

extractPerf :: OpAmp -> DF.DataFrame T.Tensor -> [(String, DF.DataFrame T.Tensor)]
            -> DF.DataFrame T.Tensor
extractPerf opAmp params nut = DF.union params perf
  where
    vdd  = T.split 1 (T.Dim 0) $ params ?? "vdd"
    perf = DF.concat $ xp opAmp vdd nut

simulate :: OpAmps -> Parameters -> IO ()
simulate opAmps params = traverseConcurrently_ (ParN n) (uncurry run) $! zip opAmps params' 
  where
    n         = fromIntegral $ length opAmps
    params'   = transposeParams params 
    run !o !p = setParams' o p >> runAll_ (session o)

readNuts :: OpAmps -> [DF.DataFrame T.Tensor] -> IO [DF.DataFrame T.Tensor]
readNuts opAmps params = do
    !nuts' <- force <$!> traverseConcurrently (ParN n') r' opAmps
    let !nuts = map (map (\(n,p) -> (n, DF.fromNutPlot p))) nuts'
        !perf = zipWith3 extractPerf opAmps params nuts
    pure $! perf
  where
    n' = fromIntegral $ length opAmps
    r' = N.readFile . (++ "/hspectre.raw") . dir . session

readResults :: OpAmps -> [Parameters] -> IO [Parameters]
readResults opAmps params = traverseConcurrently (ParN n') (uncurry read')
                          $ zip opAmps params
  where
    n'      = fromIntegral $ length opAmps
    read' o@OpAmp{..} p = do
        !nut <- N.readFile (dir session ++ "/hspectre.raw")
        let p'      = reverse $ transposeParams p
        let !perfs' = zipWith (extractPerformance o) p' $! chunksOf 10 nut
            -- perfs   = M.map V.reverse $ M.unionsWith (V.++) perfs'
            perfs   = M.map V.reverse $ M.unionsWith (V.++) perfs'
            !res    = M.unions [p, perfs]
        putStrLn $ "Calculated " ++ show (length perfs) ++ " Performances. "
        pure res

    -- read' o@OpAmp{..} p = M.unionsWith (V.++) . zipWith (extractPerformance o) p
    --                     . chunksOf 10 <$!> N.readFile (dir session ++ "/hspectre.raw")

-- | Close Spectre session
close' :: OpAmp -> IO ()
close' = stopSession . session

close :: OpAmps -> IO ()
close = mapM_ close'
