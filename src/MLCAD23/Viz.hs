{-# OPTIONS_GHC -Wall -fplugin Debug.Breakpoint #-}

{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DuplicateRecordFields #-}

-- | Design space Visualization
module MLCAD23.Viz where

import           Control.Monad
import           Data.List                 (isSuffixOf, partition, intercalate)
import           Data.Char                 (toLower)
import qualified Torch              as T
import qualified Torch.Extensions   as T
import           Data.Frame                ((??))
import qualified Data.Frame         as DF

import qualified MLCAD23.Config.CKT as CKT
import qualified MLCAD23.Config.PDK as PDK

import qualified Graphics.Plotly    as Plt

import MLCAD23.Viz.Opt
import MLCAD23.Internal

asCols :: DF.DataFrame T.Tensor -> [[Double]]
asCols = map T.asValue . T.asCols . T.toDType T.Double . DF.values

visualizeData :: PDK.ID -> CKT.ID -> IO ()
visualizeData pdk ckt = do
    df' <- DF.fromFile dataPath
    let df = DF.dropNan
           $ DF.union (T.trafo maskX <$> DF.lookup dps df')
                      (T.trafo maskY <$> DF.lookup pps df')
        dfF = DF.rowFilter (T.gt (df ?? "sr_f") 5.5) df

    dfS  <- DF.lookup pps <$> DF.sampleIO num False dfF
    cols <- optimizeOrder dfS

    let pd = asCols $ DF.lookup cols dfS
    Plt.plot plotPath $ Plt.parcoord'' cols [pd] cfg
    pure ()
  where
    num        = 1000
    dataPath   = "./data/" ++ show ckt ++ "-" ++ show pdk ++ ".pt"
    plotPath   = "./plots/" ++ show pdk ++ "/" ++ show ckt ++ "/pcp.html"
    -- (vds, dps) = partition (isSuffixOf "_vds") $ CKT.electricalParams ckt
    dps        = filter (not . isSuffixOf "_vds")  $ CKT.electricalParams ckt
    pps        = [ "a_0", "ugbw", "pm", "gm", "sr_r", "psrr_p", "psrr_n"
                     , "cmrr", "sr_f", "voff_stat", "idd"] -- , "v_oh", "v_ol" ]
    maskX      = T.toDevice T.cpu
               $ T.boolMask' (filter (isSuffixOf "_fug") dps) dps
    maskY      = T.toDevice T.cpu
               $ T.boolMask' ["sr_r", "sr_f", "ugbw"] pps
    -- params     = dps ++ pps
    cfg        = Plt.defaultConfig { Plt.title' = ""
                                   , Plt.width  = 1500
                                   , Plt.height = 600 }

visualizeModel :: PDK.ID -> CKT.ID -> IO ()
visualizeModel pdk ckt = do
    model <- T.unTraceModel <$> T.loadInferenceModel tracePath
    x <- DF.lookup (dps ++ tps) <$> CKT.randomBehaviour' ckt num True
    let y = DF.DataFrame (pps ++ vds) . model $ DF.values x

    let pd = asCols . DF.lookup (tps ++ pps) $ DF.union x y
    Plt.plot plotPath $ Plt.parcoord'' (tps ++ pps) [pd] cfg
    pure ()
  where
    num        = 1000
    tracePath  = "./models/" ++ show ckt ++ "-" ++ show pdk ++ "-trace.pt"
    plotPath   = "./plots/" ++ show pdk ++ "/" ++ show ckt ++ "/pcp.html"
    (vds, dps) = partition (isSuffixOf "_vds") $ CKT.electricalParams ckt
    tps        = ["vdd", "i0", "cl", "temp"]
    pps        = [ "a_0", "ugbw", "pm", "gm", "sr_r", "psrr_p", "psrr_n"
                 , "cmrr", "sr_f", "voff_stat", "idd", "v_oh", "v_ol" ]
    cfg        = Plt.defaultConfig { Plt.title' = ""
                                   , Plt.width  = 1500
                                   , Plt.height = 600 }

compareCircuits :: PDK.ID -> [CKT.ID] -> IO ()
compareCircuits pdk ckts = do
    models <- mapM (fmap T.unTraceModel . T.loadInferenceModel) tracePaths
    xs <- zipWithM (\p c -> DF.lookup p <$> CKT.randomBehaviour c num) dps ckts
    let ys = map fltr $ zipWith (\m x -> DF.DataFrame pps . m $ DF.values x) models xs
    
    let pd = map asCols ys
    Plt.plot plotPath $ Plt.parcoord'' pps pd cfg
    pure ()
  where
    num        = 1000
    tracePaths = [ "./models/" ++ show ckt ++ "-" ++ show pdk ++ "-trace.pt" | ckt <- ckts ]
    plotPath   = "./plots/" ++ show pdk ++ "/" ++ intercalate "-" (map show ckts) ++ "-pcp.html"
    dps        = map (filter (not . isSuffixOf "_vds") . CKT.electricalParams) ckts
    pps        = [ "a_0", "ugbw", "pm", "gm", "sr_r", "psrr_p", "psrr_n"
                 , "cmrr", "sr_f", "voff_stat", "idd", "v_oh", "v_ol" ]
    cfg        = Plt.defaultConfig { Plt.title' = ""
                                   , Plt.width  = 1500
                                   , Plt.height = 600 }
    fltr df    = DF.rowFilter ( T.logicalAnd (T.gt (df ?? "a_0")  0)
                              . T.logicalAnd (T.gt (df ?? "cmrr") 0)
                              . T.logicalAnd (T.gt (df ?? "psrr_p") 0)
                              . T.logicalAnd (T.gt (df ?? "psrr_n") 0)
                              . T.logicalAnd (T.gt (df ?? "voff_stat") 0)
                              . T.logicalAnd (T.gt (df ?? "idd") 0)
                              . T.logicalAnd (T.gt (df ?? "v_oh") 0)
                              . T.logicalAnd (T.gt (df ?? "v_ol") 0)
                              . T.logicalAnd (T.gt (df ?? "gm")  (-100))
                              $ T.logicalAnd (T.lt (df ?? "gm")   0)
                                             (T.gt (df ?? "pm")   0)
                              ) df

visualizeDeviceData :: PDK.ID -> CKT.DeviceType -> IO ()
visualizeDeviceData pdk dev = do
    df' <- DF.fromFile dataPath
    let df'' = DF.insert ["vovd"] (T.sub (df' ?? "vgs") (df' ?? "vth")) df'
        df   = DF.dropNan $ T.trafo mask <$> DF.rowFilter (fltr df'') (DF.lookup pps df'')

    dfS <- DF.sampleIO num False df
    -- cols <- optimizeOrder dfS

    let pd = asCols $ DF.lookup pps dfS

    Plt.plot plotPath $ Plt.parcoord'' pps [pd] cfg
    pure ()
  where

    num      = 10000
    dataPath = "../prehsept/data/" ++ show pdk ++ "-" ++ map toLower (show dev) ++ ".pt"
    plotPath = "./plots/" ++ show pdk ++ "/" ++ show dev ++ ".html"
    pps      = [ "L", "vgs", "vds", "vbs", "id","fug","gmoverid","gm","vdsat","vovd", "vth","gds","gmbs","cgs","cgd","cgb","csd","cbs","cbd"]
    mask     = T.boolMask' ["id", "fug", "gm", "gds", "gmbs","cgs","cgd","cgb","csd","cbs","cbd"] pps
    fltr d   = T.logicalOr (T.eq (d ?? "region") 1.0) 
             $ T.logicalOr (T.eq (d ?? "region") 2.0)
                           (T.eq (d ?? "region") 3.0)
    cfg      = Plt.defaultConfig { Plt.title'     = ""
                                    , Plt.colorScale = Plt.Discrete
                                    , Plt.width      = 1500
                                    , Plt.height     = 600 }

visualizeDeviceModel :: PDK.ID -> CKT.DeviceType -> IO ()
visualizeDeviceModel pdk dev = do
    model <- T.unTraceModel <$> T.loadInferenceModel tracePath
    let mins  = T.asTensor @[Float] [-6.3, 2.0, 6.0, 0.2, -1.5]
        maxs  = T.asTensor @[Float] [-4.3, 18.0, 10.5, 1.8, 0.0]
        mask  = T.asTensor @[Bool]  [True, False, True, False, False]
        mask' = T.asTensor @[Bool]  [True, True]

    x <- T.scale' mins maxs <$> T.randIO' [num, 5]
    let y  = T.trafo mask' . model $ T.trafo' mask x
        df = DF.DataFrame pps $ T.cat (T.Dim 1) [x, y]
        pd = asCols df

    Plt.plot plotPath $ Plt.parcoord'' pps [pd] cfg
    pure ()
  where
    num       = 10000
    tracePath = circusHome ++ "/pdk/" ++ show pdk ++ "/nmos.pt"
    plotPath  = "./plots/" ++ show pdk ++ "/" ++ show dev ++ "-model.html"
    pps       = [ "id", "gmoverid", "fug", "Vds", "Vbs", "W", "L" ]
    cfg       = Plt.defaultConfig { Plt.title' = ""
                                    , Plt.width  = 1500
                                    , Plt.height = 600 }
