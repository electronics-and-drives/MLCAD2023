{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-} 
{-# LANGUAGE OverloadedStrings #-} 
{-# LANGUAGE DuplicateRecordFields #-}

-- | Internals of MLCAD23 for performance extraction
module MLCAD23.Internal where

import           Data.Complex
import           Data.Map                         ((!))
import qualified Data.Map                  as M
import           Data.Maybe
import           Data.NutMeg                      (RealPlot, ComplexPlot)
import           Data.Vector.Storable             ((!?))
import qualified Data.Vector.Storable      as V
import qualified Data.Text                 as TXT
import qualified Data.Text.IO              as TIO
import           System.Directory                 (getHomeDirectory, copyFile)
import           System.Environment               (lookupEnv)
import           System.IO.Temp                   (createTempDirectory)
import           System.IO.Unsafe                 (unsafePerformIO)
import           System.Posix.User                (getLoginName)
import           Spectre.Interactive

import           Data.Frame                       ((??))
import qualified Data.Frame                as DF
import qualified Torch                     as T
import qualified Torch.Functional.Internal as T (real, flip)
import qualified Torch.Extensions          as T

import           MLCAD23.Util
import qualified MLCAD23.Config.PDK        as PDK
import qualified MLCAD23.Config.CKT        as CKT
import           Paths_MLCAD23


-- | Vector indexing with default
safeIndex :: (V.Storable a, Fractional a) => a -> V.Vector a -> Int -> a
safeIndex def vec idx = fromMaybe def (vec !? idx)

-- | Vector indexing with default NaN
safeIndex' :: (V.Storable a, Fractional a) => V.Vector a -> Int -> a
safeIndex' = safeIndex (0/0)

safeSelect :: T.Tensor -> T.Tensor -> T.Tensor
safeSelect t idx | 0 < T.size 0 idx = T.indexSelect 0 idx t
                 | otherwise        = T.asTensor @[Float] [0/0]

-- | Circus Home DirectFuacide
circusHome' :: IO FilePath
circusHome' = do
    home' <- getHomeDirectory
    fromMaybe (home' ++ "/.serafin") <$> lookupEnv "CIRCUS_HOME" 

-- | Circus Home Directory
{-# NOINLINE circusHome #-}
circusHome :: FilePath
circusHome = unsafePerformIO circusHome'

-- | Lines starting with `include` in netlist
includeLine :: PDK.Include -> TXT.Text
includeLine PDK.Include{..} = TXT.pack $ inc ++ " " ++ sec
  where
    inc = "include \"" ++ path ++ "\""
    sec = "section=" ++ section

-- | elements of lines starting with `parameters` in netlist
paramSpec :: String -> Double -> TXT.Text
paramSpec param value = TXT.pack $ param ++ "=" ++ show value

-- | lines starting with `parameters` in netlist
paramLine :: [TXT.Text] -> TXT.Text
paramLine = TXT.unwords . ("parameters" :)

-- | Setup Temporary directory for netlist and simulation results
setupDir :: CKT.ID -> PDK.ID -> CKT.CFG -> PDK.CFG -> FilePath -> IO FilePath
setupDir cktId pdkId cktCfg pdkCfg netlist = do
    usr <- getLoginName
    net <- TIO.readFile netlist

    let subckt  = TXT.unlines [includes, tbParams, opParams, aeParams, net, saves]
        tmplate = usr ++ "-" ++ show cktId ++ "-" ++ show pdkId

    tmpDir <- createTempDirectory "/tmp" tmplate 

    TIO.writeFile (tmpDir ++ "/op.scs") subckt

    getDataFileName "rsrc/testbench.scs" >>= flip copyFile (tmpDir ++ "/tb.scs")

    pure tmpDir
  where
    includes  = TXT.unlines . map includeLine . PDK.include $ pdkCfg
    defaults  = M.union (CKT.testbench . CKT.parameters $ cktCfg)
                        (PDK.testbench pdkCfg)
    tbParams  = paramLine . map (uncurry paramSpec)
                          $ M.toList defaults
    opParams  = paramLine . map (uncurry paramSpec)
                          . M.toList . CKT.geometrical 
                          . CKT.parameters $ cktCfg
    area      = TXT.pack . CKT.area . CKT.parameters $ cktCfg
    aeParams  = paramLine [TXT.concat ["parameters area=", area]]
    opPre     = TXT.pack $ PDK.prefix (PDK.dcop . PDK.devices $ pdkCfg :: PDK.DCOPConfig)
    opSuf     = TXT.pack $ PDK.suffix (PDK.dcop . PDK.devices $ pdkCfg :: PDK.DCOPConfig)
    opPar     = map TXT.pack . PDK.parameters . PDK.dcop . PDK.devices $ pdkCfg
    saves     = TXT.intercalate "\t\\\n\t" 
              $ "save " : [ TXT.concat ["\t", opPre, "*", opSuf, ":", param] 
                          | param <- opPar ]

-- | Performance Extraction Result
type Parameters  = M.Map String (V.Vector Double)
type Parameters' = M.Map String Double

-- | Extract DC Match configuration
dcmConfig :: PDK.DCMConfig' -> CKT.DeviceType -> [PDK.DCMConfig]
dcmConfig PDK.DCMConfig'{..} CKT.NMOS = nmos
dcmConfig PDK.DCMConfig'{..} CKT.PMOS = pmos
dcmConfig _                  _        = undefined

-- | All Performance Parameters
performanceParameters :: M.Map String String
performanceParameters = M.fromList [ ("area",      "Estimated Area")
                                   , ("a_0",       "DC Loop Gain")
                                   , ("cmrr",      "Common Mode Rejection Ratio")
                                   , ("cof",       "Cross-Over Frequency")
                                   , ("gm",        "Gain Margin")
                                   , ("i_out_max", "Maximum output Current")
                                   , ("i_out_min", "Minimum output Current")
                                   , ("idd",       "Current Consumption")
                                   , ("iss",       "Current Consumption")
                                   , ("os_f",      "Overshoot Falling")
                                   , ("os_r",      "Overshoot Rising")
                                   , ("pm",        "Phase Margin")
                                   , ("psrr_n",    "Power Supply Rejection Ratio")
                                   , ("psrr_p",    "Power Supply Rejection Ratio")
                                   , ("sr_f",      "Slew Rate Falling")
                                   , ("sr_r",      "Slew Rate Rising")
                                   , ("ugbw",      "Unity Gain Bandwidth")
                                   , ("v_ih",      "Input Voltage Hight")
                                   , ("v_il",      "Input Voltage Low")
                                   , ("v_oh",      "Output Voltage High")
                                   , ("v_ol",      "Output Voltage Low")
                                   , ("vn_1Hz",    "Output Referred Noise @ 1Hz")
                                   , ("vn_10Hz",   "Output Referred Noise @ 10Hz")
                                   , ("vn_100Hz",  "Output Referred Noise @ 100Hz")
                                   , ("vn_1kHz",   "Output Referred Noise @ 1kHz")
                                   , ("vn_10kHz",  "Output Referred Noise @ 10kHz")
                                   , ("vn_100kHz", "Output Referred Noise @ 100kHz")
                                   , ("voff_stat", "Statistical Offset")
                                   , ("voff_syst", "Systematic Offset") ]

-- | Estimate Area given the current sizing
estimatedArea' :: Session -> IO Parameters
estimatedArea' session = do
    area <- V.singleton <$> getParameter session "area"
    pure $ M.fromList [ ("area", V.force area) ]

-- | Unsafe area estimation
{-# NOINLINE estimatedArea #-}
estimatedArea :: Session -> Parameters
estimatedArea = unsafePerformIO . estimatedArea'

offset' :: DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor
offset' = DF.rename ["voff_stat", "voff_syst"]
        . DF.lookup ["totalOutput.sigmaOut", "totalOutput.dcOp"]

-- | Extract performances from `dcmatch` analysis
offset :: RealPlot -> Parameters
offset plot = M.fromList [ ("voff_stat", V.force voffStat)
                         , ("voff_syst", V.force voffSyst) ]
  where
    voffStat = plot ! "totalOutput.sigmaOut"
    voffSyst = plot ! "totalOutput.dcOp"

stability' :: DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor
stability' plot = DF.DataFrame ["a_0", "ugbw", "pm", "gm", "cof"]
                $ T.stack (T.Dim 1) [a0db, ugbw, pm, gm, cof]
  where
    loopGain = plot ?? "loopGain"
    freq     = T.real $ plot ?? "freq"
    gain     = T.db20 loopGain
    phas     = T.phase loopGain
    a0Idx    = T.indexSelect' 0 [0] . T.squeezeAll . T.nonzero $ T.le gain 0.0
    p0Idx    = T.indexSelect' 0 [0] . T.squeezeAll . T.nonzero $ T.le phas 0.0
    a0db     = T.indexSelect' 0 [0] gain
    ugbw     = T.indexSelect 0 a0Idx freq
    pm       = T.indexSelect 0 a0Idx phas
    cof      = T.indexSelect 0 p0Idx freq
    gm       = T.indexSelect 0 p0Idx gain

-- | Extract performances from `stb` analysis
stability :: ComplexPlot -> Parameters
stability plot = M.fromList [ ("a_0",  V.force a0db) 
                            , ("ugbw", V.force f0db)
                            , ("pm",   V.force pm)
                            , ("gm",   V.force gm)
                            , ("cof",  V.force cof) ]
  where
    loopGain = plot ! "loopGain"
    freq     = V.map realPart $ plot ! "freq"
    gain     = V.map db20 loopGain
    phase'   = V.map angle loopGain
    a0Idx    = idxFirstOf Falling 0.0 gain
    ph0Idx   = idxFirstOf Falling 0.0 phase'
    a0db     = V.slice 0 1 gain
    f0db     = V.slice a0Idx 1 freq
    f0Idx    = idxCloseTo (V.unsafeHead f0db) freq
    -- f0Idx    = fromJust $ V.elemIndex f0db freq
    pm       = V.slice f0Idx 1 phase'
    cof      = V.slice ph0Idx 1 freq
    gm       = V.slice ph0Idx 1 gain

transient' :: Float -> DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor
transient' vs plot = DF.DataFrame ["sr_r", "sr_f", "os_r", "os_f"]
                   $ T.stack (T.Dim 1) [srR, srF, osR, osF]
  where
    time     = plot ?? "time"
    out      = plot ?? "out"
    idx100   = T.toInt . snd . T.minDim (T.Dim 0) T.KeepDim . T.abs $ 100.0e-9 - time
    idx090   = T.toInt . snd . T.minDim (T.Dim 0) T.KeepDim . T.abs $ 90.0e-6  - time
    idx050   = T.toInt . snd . T.minDim (T.Dim 0) T.KeepDim . T.abs $ 50.0e-6  - time
    idx099   = T.toInt . snd . T.minDim (T.Dim 0) T.KeepDim . T.abs $ 99.9e-6  - time
    rising   = T.sliceDim 0 idx100 (idx050 - idx100) 1 out
    falling  = T.sliceDim 0 (idx050 + 1) (idx099 - idx050 - 1) 1 out
    risingT  = T.sliceDim 0 idx100 (idx050 - idx100) 1 time
    fallingT = T.sliceDim 0 (idx050 + 1) (idx099 - idx050 - 1) 1 time
    lower    = T.asTensor $ (0.1 * vs) - (vs / 2.0)
    upper    = T.asTensor $ (0.9 * vs) - (vs / 2.0)
    r1       = safeSelect risingT . T.squeezeAll . T.nonzero $ T.ge rising lower
    r2       = safeSelect risingT . T.squeezeAll . T.nonzero $ T.ge rising upper
    deltaR   = r2 - r1
    srR      = T.toDType T.Float (T.gt deltaR 0.0) * ((upper - lower) / deltaR)
    fSize    = T.asTensor $ T.size 0 falling
    f1Idx    = T.subScalar @Int 1 . T.sub fSize . T.indexSelect' 0 [0] . T.squeezeAll
             . T.nonzero . T.le upper $ T.flip falling [0]
    fWin     = T.sliceDim 0 (T.toInt f1Idx) (T.size 0 falling) 1 falling
    f2Idx    = T.add f1Idx . T.indexSelect' 0 [0] . T.squeezeAll . T.nonzero $ T.le fWin lower
    f1       = safeSelect fallingT f1Idx
    f2       = safeSelect fallingT f2Idx
    deltaF   = f2 - f1
    srF      = T.toDType T.Float (T.gt deltaF 0.0) * ((upper - lower) / deltaF)
    osR      = T.mulScalar @Float 100.0 
             . T.div (T.sub (T.maxDim' (T.Dim 0) rising) (T.indexSelect' 0 [idx050] out))
             $ T.sub (T.indexSelect' 0 [idx050] out) (T.indexSelect' 0 [idx100] out)
    osF      = T.mulScalar @Float 100.0 
             . T.div (T.sub (T.minDim' (T.Dim 0) falling) (T.indexSelect' 0 [idx090] out))
             $ T.sub (T.indexSelect' 0 [idx090] out) (T.indexSelect' 0 [idx050] out)

-- | Extract performances from `tran` analysis
transient :: Double -> RealPlot -> Parameters
transient vs plot = M.fromList [ ("sr_r", V.force $ V.fromList [rSR])
                               , ("sr_f", V.force $ V.fromList [fSR])
                               , ("os_r", V.force $ V.fromList [rOS])
                               , ("os_f", V.force $ V.fromList [fOS]) ]
  where
    time     = plot ! "time"
    out      = plot ! "OUT"
    idx100   = idxCloseTo 100.0e-9 time
    idx090   = idxCloseTo 90.0e-6  time
    idx050   = idxCloseTo 50.0e-6  time
    idx099   = idxCloseTo 99.9e-6  time
    rising   = V.slice idx100 (idx050 - idx100) out
    falling  = V.slice (idx050 + 1) (idx099 - idx050 - 1) out
    risingT  = V.slice idx100 (idx050 - idx100) time
    fallingT = V.slice (idx050 + 1) (idx099 - idx050 - 1) time
    lower    = (0.1 * vs) - (vs / 2.0)
    upper    = (0.9 * vs) - (vs / 2.0)
    r1       = safeIndex' risingT $ idxFirstOf Rising lower rising
    r2       = safeIndex' risingT $ idxFirstOf Rising upper rising
    rDelta   = r2 - r1
    rSR      = if rDelta > 0.0 then (upper - lower) / rDelta else 0.0
    f1Idx    = subtract 1 . (V.length falling -) . idxFirstOf Rising upper
             $ V.reverse falling
    fWin     = V.drop f1Idx falling
    f2Idx    = f1Idx + idxFirstOf Falling lower fWin
    f1       = safeIndex' fallingT f1Idx
    f2       = safeIndex' fallingT f2Idx
    fDelta   = f2 - f1
    fSR      = if fDelta > 0.0 then (lower - upper) / fDelta else 0.0
    rOS      = 100.0 * (V.maximum rising  - safeIndex' out idx050)
             / (safeIndex' out idx050 - safeIndex' out idx100)
    fOS      = 100.0 * (V.minimum falling - safeIndex' out idx090)
             / (safeIndex' out idx090 - safeIndex' out idx050)

outputReferredNoise' :: DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor
outputReferredNoise' plot = DF.DataFrame cols vals
  where
    cols = ["vn_1Hz", "vn_10Hz", "vn_100Hz", "vn_1kHz", "vn_10kHz", "vn_100kHz"]
    fs   = [1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5]
    freq = plot ?? "freq"
    out  = plot ?? "out"
    idx  = T.cat (T.Dim 0) $ map (T.findClosestIdx freq) fs
    vals = T.indexSelect 0 idx out

-- | Extract performances from `noise` analysis
outputReferredNoise :: RealPlot -> Parameters
outputReferredNoise plot = M.fromList $ zip cols vals
  where
    cols   = ["vn_1Hz", "vn_10Hz", "vn_100Hz", "vn_1kHz", "vn_10kHz", "vn_100kHz"]
    fs     = [1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5]
    freq   = plot ! "freq"
    out    = plot ! "out"
    frqIdx = map (`idxCloseTo` freq) fs
    vals   = map (V.force . V.singleton . (out V.!)) frqIdx

outSwingDC' :: Float -> T.Tensor -> DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor
outSwingDC' dev vdd plot = DF.DataFrame ["v_oh", "v_ol"]
                         $ T.stack (T.Dim 1) [vOH, vOL]
  where
    out      = plot ?? "OUT"
    outIdeal = plot ?? "OUT_IDEAL"
    vid      = plot ?? "vid"
    out0     = T.indexSelect 0 (T.findClosestIdx vid 0.0) out
    outDC    = T.sub out out0
    devRel   = T.abs (outDC - outIdeal) / vdd
    mVil     = T.toDType T.Float $ T.le vid 0.0
    mVih     = T.toDType T.Float $ T.ge vid 0.0
    infl     = (1.0 - mVil) * T.asTensor @Float (1/0)
    infh     = (1.0 - mVih) * T.asTensor @Float (1/0)
    vilIdx   = snd . T.minDim (T.Dim 0) T.KeepDim . T.add infl
             $ T.abs (T.mulScalar dev  mVil) - (devRel * mVil)
    vihIdx   = snd . T.minDim (T.Dim 0) T.KeepDim . T.add infh
             $ T.abs (T.mulScalar dev  mVih) - (devRel * mVih)
    vOH      = T.add (vdd / 2.0) $ T.indexSelect 0 vihIdx outDC
    vOL      = T.add (vdd / 2.0) $ T.indexSelect 0 vilIdx outDC

-- | Extract performances from `dc1` analysis
outSwingDC :: Double -> Double -> RealPlot -> M.Map String (V.Vector Double)
outSwingDC vdd dev plot = M.fromList [ ("v_oh", V.force vohDC)
                                     , ("v_ol", V.force volDC) ]
  where
    out      = plot ! "OUT"
    outIdeal = plot ! "OUT_IDEAL"
    vid      = plot ! "vid"
    out0     = safeIndex' out $ idxCloseTo 0.0 vid
    outDC    = V.map (out0 `subtract`) out
    devRel   = V.zipWith (\a b -> (/vdd) . abs $ a - b) outDC outIdeal
    vil      = V.map (abs . (dev -) . (devRel V.!)) $ V.findIndices (<=0.0) vid 
    vilIdx   = V.minIndex vil
    vih      = V.map (abs . (dev -) . (devRel V.!)) $ V.findIndices (>=0.0) vid 
    vihIdx   = V.minIndex vih
    volDC    = V.map ((vdd / 2.0) +) $ V.slice vilIdx 1 outDC
    vohDC    = V.map ((vdd / 2.0) +) $ V.slice vihIdx 1 outDC

rejection' :: DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor
rejection' plot = DF.DataFrame ["psrr_p", "psrr_n", "cmrr"]
                $ T.stack (T.Dim 0) [psrrp, psrrn, cmrr]
  where
    freq    = T.real $ plot ?? "freq"
    vidDB   = T.db20 $ plot ?? "VID"
    vicmDB  = T.db20 $ plot ?? "VICM"
    vsuppDB = T.db20 $ plot ?? "VSUPP"
    vsupnDB = T.db20 $ plot ?? "VSUPN"
    fminIdx = snd $ T.minDim (T.Dim 0) T.KeepDim freq
    psrrp   = T.indexSelect 0 fminIdx $ vidDB - vsuppDB
    psrrn   = T.indexSelect 0 fminIdx $ vidDB - vsupnDB
    cmrr    = T.indexSelect 0 fminIdx $ vidDB - vicmDB

-- | Extract performances from `xf` analysis
rejection :: ComplexPlot -> Parameters
rejection plot = M.fromList [ ("psrr_p", V.force psrrp)
                            , ("psrr_n", V.force psrrn)
                            , ("cmrr",   V.force cmrr) ]
  where
    freq    = V.map realPart $ plot ! "freq"
    vidDB   = V.map db20     $ plot ! "VID"
    vicmDB  = V.map db20     $ plot ! "VICM"
    vsuppDB = V.map db20     $ plot ! "VSUPP"
    vsupnDB = V.map db20     $ plot ! "VSUPN"
    fMinIdx = V.minIndex freq
    psrrp   = V.singleton . (V.! fMinIdx) $ V.zipWith (-) vidDB vsuppDB
    psrrn   = V.singleton . (V.! fMinIdx) $ V.zipWith (-) vidDB vsupnDB
    cmrr    = V.singleton . (V.! fMinIdx) $ V.zipWith (-) vidDB vicmDB
    -- psrrp   = V.slice 0 1 $ V.zipWith (-) vidDB vsuppDB
    -- psrrn   = V.slice 0 1 $ V.zipWith (-) vidDB vsupnDB
    -- cmrr    = V.slice 0 1 $ V.zipWith (-) vidDB vicmDB

outSwingAC' :: T.Tensor -> T.Tensor -> DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor
outSwingAC' vdd a3db plot = DF.DataFrame ["v_ih", "v_il"]
                          $ T.stack (T.Dim 1) [vIH, vIL]
  where
    vicm    = T.real $ plot ?? "vicm"
    outAC   = T.db20 $ plot ?? "OUT"
    leq0m   = T.toDType T.Float $ T.le vicm 0.0
    geq0m   = T.toDType T.Float $ T.ge vicm 0.0
    leq0    = T.mul outAC . T.add (vicm * leq0m) . T.mul (1.0 - leq0m)
            $ T.asTensor @Float (1/0)
    geq0    = T.mul outAC . T.add (vicm * geq0m) . T.mul (1.0 - geq0m)
            $ T.asTensor @Float (1/0)
    leq0Idx = snd . T.minDim (T.Dim 1) T.KeepDim . T.abs $ a3db - leq0
    vIL     = if T.any (T.le vicm 0.0)
                 then T.add (vdd / 2.0) $ T.indexSelect 0 leq0Idx vicm
                 else T.zeros' [1] 
    geq0Idx = snd . T.minDim (T.Dim 1) T.KeepDim . T.abs $ a3db - geq0
    vIH     = if T.any (T.ge vicm 0.0)
                 then T.add (vdd / 2.0) $ T.indexSelect 0 geq0Idx vicm
                 else T.zeros' [1] 

-- | Extract performances from `ac` analysis
outSwingAC :: Double -> Double -> ComplexPlot -> Parameters
outSwingAC a3db vdd plot = M.fromList [ ("v_ih", V.force vihAC)
                                      , ("v_il", V.force vilAC) ]
  where
    vicm    = V.map realPart $ plot ! "vicm"
    outAC   = V.map db20     $ plot ! "OUT"
    leq0    = V.map (outAC V.!) $ V.findIndices (<= 0.0) vicm
    geq0    = V.map (outAC V.!) $ V.findIndices (>= 0.0) vicm
    leq0Idx = V.minIndex $ V.map (abs . (a3db-)) leq0
    vilAC   = if V.any (<= 0.0) vicm
                 then V.map ((vdd/2.0)+) $ V.slice leq0Idx 1 vicm
                 else V.fromList [0.0]
    geq0Idx = V.minIndex $ V.map (abs . (a3db-)) geq0
    vihAC   = if V.any (>= 0.0) vicm
                 then V.map ((vdd/2.0)+) $ V.slice geq0Idx 1 vicm
                 else V.fromList [0.0]

outputCurrent' :: DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor
outputCurrent' dc3 dc4 = DF.DataFrame ["i_out_max", "i_out_min"]
                       $ T.stack (T.Dim 1) [iOutMax, iOutMin]
  where
    iOutMin = dc3 ?? "DUT:O"
    iOutMax = dc4 ?? "DUT:O"

-- | Extract performances from `dc3` and `dc4` analysis
outputCurrent :: RealPlot -> RealPlot -> Parameters
outputCurrent dc3 dc4 = M.fromList [ ("i_out_max", V.force iOutMax)
                                   , ("i_out_min", V.force iOutMin) ]
  where
    iOutMin = dc3 ! "DUT:O"
    iOutMax = dc4 ! "DUT:O"

operatingPoint' :: M.Map String String -> DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor
operatingPoint' params plot = DF.rename (M.elems colIds) $ DF.lookup (M.keys colIds) plot
  where
    colIds = M.union params $ M.fromList [ ("DUT:VDD", "idd"), ("DUT:VSS", "iss") ]

-- | Extract performances from `dcop` analysis
operatingPoint :: M.Map String String -> RealPlot -> M.Map String (V.Vector Double)
operatingPoint params = M.mapKeys (colIds !) 
                      . flip M.restrictKeys (M.keysSet colIds) . M.map V.force
  where
    colIds = M.union params $ M.fromList [ ("DUT:VDD", "idd"), ("DUT:VSS", "iss") ]
