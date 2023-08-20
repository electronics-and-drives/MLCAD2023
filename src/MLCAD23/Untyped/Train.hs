{-# OPTIONS_GHC -Wall -fplugin Debug.Breakpoint #-}

{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Statically Typed Model Training
module MLCAD23.Untyped.Train where

import qualified Torch                         as T
import qualified Torch.Extensions              as T
import qualified Torch.Optim.CppOptim          as T
import qualified Torch.Functional.Internal     as T   (mish)
-- import           Data.Frame                           ((??))
import qualified Data.Frame                    as DF
import           MLCAD23.Util

import           Data.Default.Class
import           Control.Monad.State
import           System.Directory                     (createDirectoryIfMissing)
import           Data.List                            (isSuffixOf, partition)
import           GHC.Generics
import           GHC.Float                            (float2Double)

import qualified MLCAD23.Config.CKT            as CKT
import qualified MLCAD23.Config.PDK            as PDK
-- import qualified Graphics.Vega.VegaLite.Simple as VL
-- import qualified Graphics.Vega.VegaLite.View   as VL
import qualified Graphics.Plotly               as Plt

-- | A single batch of Data
data Batch = Batch { xs :: !T.Tensor -- ^ Input Data
                   , ys :: !T.Tensor -- ^ Output Data
                   } deriving (Show, Eq)

-- | Type Alias for Adam C++ Optimizer
type Optim = T.CppOptimizerState T.AdamOptions
-- type Optim = T.Adam

-- | Untyped Model Specification
data ModelSpec = ModelSpec { numX :: !Int                    -- ^ Number of input neurons
                           , numY :: !Int                    -- ^ Number of output neurons
                           , actF :: !(T.Tensor -> T.Tensor) -- ^ Activation Function
                           } -- deriving (Show, Eq)

-- | Untyped Network Architecture
data Model = Model { l0 :: !T.Linear
                   , l1 :: !T.Linear
                   , l2 :: !T.Linear
                   , l3 :: !T.Linear
                   , l4 :: !T.Linear
                   , l5 :: !T.Linear
                   , l6 :: !T.Linear
                   , l7 :: !T.Linear
                   , φ  :: !(T.Tensor -> T.Tensor)
                   } deriving (Generic, T.Parameterized)

-- | Untyped Neural Network Weight initialization
instance T.Randomizable ModelSpec Model where
    sample ModelSpec{..} = Model <$> T.sample (T.LinearSpec   numX 64)
                                 <*> T.sample (T.LinearSpec   64   128)
                                 <*> T.sample (T.LinearSpec   128  256)
                                 <*> T.sample (T.LinearSpec   256  512)
                                 <*> T.sample (T.LinearSpec   512  256)
                                 <*> T.sample (T.LinearSpec   256  128)
                                 <*> T.sample (T.LinearSpec   128  64)
                                 <*> T.sample (T.LinearSpec   64   numY)
                                 <*> pure actF

-- | Untyped Neural Network Forward Pass with scaled Data
forward :: Model -> T.Tensor -> T.Tensor
forward Model{..} = T.linear l7 . φ
                  . T.linear l6 . φ
                  . T.linear l5 . φ
                  . T.linear l4 . φ
                  . T.linear l3 . φ
                  . T.linear l2 . φ
                  . T.linear l1 . φ
                  . T.linear l0

-- | Remove Gradient for tracing / scripting
noGrad :: (T.Parameterized f) => f -> IO f
noGrad net = do
    params <- mapM detachToCPU (T.flattenParameters net) 
                >>= mapM (`T.makeIndependentWithRequiresGrad` False)
    pure $ T.replaceParameters net params
  where
    detachToCPU = T.detach . T.toDevice T.cpu . T.toDependent

-- | Don't Detach!
withGrad :: (T.Parameterized f) => f -> IO f
withGrad net = T.replaceParameters net 
                <$> mapM ( (`T.makeIndependentWithRequiresGrad` False) 
                           . T.toDevice T.cpu . T.toDependent )
                         (T.flattenParameters net)

-- | Save Model and Optimizer Checkpoint
saveCheckPoint :: (T.Parameterized f) => FilePath -> f -> IO ()
saveCheckPoint path net = T.saveParams net path

-- | Load a Saved Model and Optimizer CheckPoint
loadCheckPoint :: (T.Randomizable spec b, T.Parameterized b) => FilePath
               -> spec -> IO b
loadCheckPoint path spec = T.sample spec >>= (`T.loadParams` path)

-- | Training State
data TrainState = TrainState { epoch        :: !Int            -- ^ Current Epoch
                             , lossBuffer   :: ![T.Tensor]     -- ^ Buffer for batch losses
                             , lossBuffer'  :: ![T.Tensor]     -- ^ Buffer for batch losses
                             , trainLoss    :: ![Float]        -- ^ Training Loss
                             , validLoss    :: ![Float]        -- ^ Validation Loss
                             , model        :: !Model          -- ^ Surrogate Model
                             , optim        :: !Optim          -- ^ Optimizier
                             , learningRate :: !T.LearningRate -- ^ Learning Rate
                             , batchSize    :: !Int            -- ^ Batch Size
                             , modelPath    :: !FilePath       -- ^ Save Path
                             , xParams      :: ![String]       -- ^ X Columns
                             , yParams      :: ![String]       -- ^ Y Columns
                             }

-- | Validation Step without gradient
validStep :: Model -> Batch -> T.Tensor
validStep net Batch{..} = T.meanDim (T.Dim 0) T.RemoveDim T.Float
                        $ T.l1Loss T.ReduceNone ys' ys
  where
    ys' = forward net xs

-- | Validation Epoch
validEpoch :: [Batch] -> StateT TrainState IO ()
validEpoch [] = do 
    s@TrainState{..} <- get
    let l' = T.stack (T.Dim 0) lossBuffer'
        l  = T.meanDim (T.Dim 0) T.RemoveDim T.Float l'
        vl = T.asValue $ T.meanAll l
    liftIO . putStrLn $ "\tValid Loss: " ++ show l
    put s { lossBuffer' = [], validLoss = vl : validLoss }
validEpoch (b:bs) = do
    s@TrainState{..} <- get
    let l = validStep model b
    put $ s {lossBuffer' = l : lossBuffer'}
    validEpoch bs 

-- | Training Step with Gradient
trainStep ::  T.LearningRate -> Model -> Optim -> Batch
          -> IO (Model, Optim, T.Tensor)
trainStep α m o Batch{..} = do
    -- ε <- T.mulScalar (1.0e-7 :: Float) 
    --         <$> T.randnIO (T.shape xs) (T.withDevice T.gpu T.defaultOpts)
    -- let ys' = forward m . T.add ε $ xs
    --     l   = T.l1Loss T.ReduceMean ys' ys
    (m', o') <- T.runStep m o l α
    pure (m', o', l)
  where
    -- l = T.mseLoss ys $ forward m xs
    l = T.l1Loss T.ReduceSum ys $ forward m xs
    -- l = T.sumAll $ T.meanDim (T.Dim 0) T.RemoveDim T.Float
    --   . T.l1Loss T.ReduceNone ys $ forward m xs
    -- l = T.mean $ T.sumDim (T.Dim 0) T.RemoveDim T.Float
    --   . T.l1Loss T.ReduceNone ys $ forward m xs

-- | Training Epoch
trainEpoch :: [Batch] -> StateT TrainState IO ()
trainEpoch   []   = do
    s@TrainState{..} <- get
    let l' = T.meanAll . T.cat (T.Dim 0) . map (T.reshape [1]) $ lossBuffer
        l  = T.asValue l'
    liftIO . putStrLn $ "\tTrain Loss: " ++ show l'
    put s {lossBuffer = [], trainLoss = l : trainLoss}
    pure ()
trainEpoch (b:bs) = do
    s@TrainState{..} <- get
    (model', optim', loss') <- liftIO $ trainStep learningRate model optim b
    put $ s { model = model', optim = optim', lossBuffer = loss' : lossBuffer }
    trainEpoch bs

-- | Data Shuffler
shuffleData :: DF.DataFrame T.Tensor -> StateT TrainState IO [Batch]
shuffleData df = do
    TrainState{..} <- get
    df' <- liftIO $ DF.shuffleIO df
    let xs' = splitBatches batchSize . DF.values $ DF.lookup xParams df'
        ys' = splitBatches batchSize . DF.values $ DF.lookup yParams df'
    pure $ zipWith Batch xs' ys'

-- | Data Shuffler
constData :: DF.DataFrame T.Tensor -> State TrainState [Batch]
constData df = do
    TrainState{..} <- get
    let xs' = splitBatches batchSize . DF.values $ DF.lookup xParams df
        ys' = splitBatches batchSize . DF.values $ DF.lookup yParams df
    pure $ zipWith Batch xs' ys'

-- | Training in State Monad
runTraining :: DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor
            -> StateT TrainState IO Model
runTraining td vd = do
    ep <- gets epoch
    liftIO . putStrLn $ "Epoch " ++ show ep ++ ":"
    shuffleData td >>= trainEpoch
    shuffleData vd >>= validEpoch
    -- tl  <- constData' td >>= trainEpoch
    -- vl  <- constData' vd >>= validEpoch'
    s@TrainState{..} <- get
    when (head validLoss == minimum validLoss) $ do
        liftIO $ putStrLn "\tNew model Saved!"
        liftIO $ saveCheckPoint (modelPath ++ "-checkpoint.pt") model
    let epoch' = epoch - 1
    put $ s {epoch = epoch'}
    if epoch' <= 0 then pure model else runTraining td vd
  -- where
  --   validEpoch' = liftState . validEpoch
  --   constData'  = liftState . constData

-- | Main Training Function
train :: Int -> CKT.ID -> PDK.ID -> PDK.Corner -> IO ()
train num ckt pdk crn = do
    df' <- DF.fromFile $ dataPath ++ show crn ++ ".pt" -- "./data/mil-xt018.pt"

    let msk  = CKT.trainMask ckt df'
        dfT  = DF.dropNan . DF.rowFilter msk
             $ DF.union (T.trafo maskX <$> DF.lookup paramsX df')
                        (T.trafo maskY <$> DF.lookup paramsY df')
        ts   = floor @Float $ 0.85 * realToFrac np
        dfX' = DF.lookup paramsX dfT
        dfY' = DF.lookup paramsY dfT
        minX = fst . T.minDim (T.Dim 0) T.RemoveDim . DF.values $ dfX'
        maxX = fst . T.maxDim (T.Dim 0) T.RemoveDim . DF.values $ dfX'
        minY = fst . T.minDim (T.Dim 0) T.RemoveDim . DF.values $ dfY'
        maxY = fst . T.maxDim (T.Dim 0) T.RemoveDim . DF.values $ dfY'
        dfX  = T.scale minX maxX <$> dfX'
        dfY  = T.scale minY maxY <$> dfY'
        df   = DF.dropNan $ DF.union dfX dfY
        np   = min np' $ DF.nRows df

    [!dfTrain,!dfValid] <- map (DF.DataFrame params) . T.split ts (T.Dim 0) . DF.values
                        <$> DF.sampleIO np False df

    putStrLn $ "Number of Training Data-Points: " ++ show (DF.nRows dfTrain)
    putStrLn $ "Number of Validation Data-Points: " ++ show (DF.nRows dfValid)

    mdl <- T.toDevice T.gpu <$> T.sample (ModelSpec dimX dimY φ')
    opt <- T.initOptimizer opt' $ T.flattenParameters mdl
    -- let opt = T.mkAdam 0 β1 β2 $ T.flattenParameters mdl

    let initialState = TrainState num l l l l mdl opt α bs modelPath paramsX paramsY
        finalPath    = modelPath ++ "-final.pt"

    -- evalStateT (runTraining dfTrain dfValid) initialState
    --     >>= withGrad >>= saveCheckPoint finalPath
    (m,s) <- runStateT (runTraining dfTrain dfValid) initialState
    withGrad m >>= saveCheckPoint finalPath

    let tl = reverse $ trainLoss s
        vl = reverse $ validLoss s

    storeLoss pdk ckt tl vl

    putStrLn $ "Final Checkpoint saved at: " ++ finalPath

    !mdl' <- loadCheckPoint (modelPath ++ "-checkpoint.pt") (ModelSpec dimX dimY φ')
                >>= withGrad

    let predict = neg maskNeg
                . T.trafo' maskY . T.scale' minY maxY
                . forward mdl'
                . T.scale minX maxX . T.trafo maskX

    putStrLn $ "\t" ++ show paramsX ++ " -> " ++ show paramsY

    testModel pdk ckt predict paramsX paramsY $ DF.rowFilter msk df'
    
    T.traceModel dimX predict >>= T.saveInferenceModel tracePath
    mdl'' <- T.unTraceModel <$> T.loadInferenceModel tracePath
    
    testModel pdk ckt mdl'' paramsX paramsY $ DF.rowFilter msk df'

    putStrLn $ "Traced Model saved at: " ++ tracePath

    -- mdl'' <- T.unTraceModel <$> T.loadInferenceModel tracePath
    -- testModel pdk ckt mdl'' paramsX paramsY $ DF.rowFilter msk df'
    -- DF.sampleIO np False df' >>= testModel pdk ckt mdl'' paramsX paramsY
  where
    dataPath       = "./data/" ++ show ckt ++ "-" ++ show pdk ++ "-"
    modelPath      = "./models/" ++ show ckt ++ "-" ++ show pdk ++ "-stat"
    tracePath      = modelPath ++ "-trace.pt"
    l              = [] -- T.toDevice T.gpu . T.asTensor $ (1.0 :: Float) / 0.0
    φ'             = T.mish
    α'             = 1.0e-4
    α              = T.asTensor α'
    β1             = 0.900
    β2             = 0.999
    β1'            = float2Double β1
    β2'            = float2Double β2
    bs             = 16
    np'            = 50000 -- 2000
    opt'           = def { T.adamLr          = α'
                         , T.adamBetas       = (β1', β2')
                         , T.adamEps         = 1.0e-8
                         , T.adamWeightDecay = 0.0
                         , T.adamAmsgrad     = False
                         } :: T.AdamOptions
    (vds, ops)     = partition (isSuffixOf "_vds") $ CKT.electricalParams ckt
    paramsY        = [ "a_0", "ugbw", "pm", "gm", "sr_r", "psrr_p", "psrr_n"
                     , "cmrr", "sr_f", "voff_stat", "idd"
                     ] ++ vds
    paramsX        = ops
    params         = paramsX ++ paramsY
    dimY           = length paramsY
    dimX           = length paramsX
    maskX          = T.toDevice T.cpu
                   $ T.boolMask' (filter (isSuffixOf "_fug") paramsX) paramsX
    maskY          = T.toDevice T.cpu
                   $ T.boolMask' ["sr_r", "sr_f", "ugbw"] paramsY
    neg m t        = T.add (T.mul (0.0 - m) t) (T.mul (1.0 - m) t)
    !maskNeg'      = T.boolMask' ["sr_f"] paramsY
    !maskNeg       = T.toDType T.Float $ T.toDevice T.cpu maskNeg'

-- | Split tensor into 
splitBatches :: Int -> T.Tensor -> [T.Tensor]
splitBatches bs = filter ((bs==) . head . T.shape) . T.split bs (T.Dim 0)
                . T.toDevice T.gpu

storeLoss :: PDK.ID -> CKT.ID -> [Float] -> [Float] -> IO ()
storeLoss pdk ckt tl' vl' = do
    createDirectoryIfMissing True base
    writeFile path csv
    Plt.plot path' $ Plt.scatter ["train", "valid"] ep [tl,vl] cfg
    pure ()
  where
    base  = "./plots/" ++ show pdk ++ "/" ++ show ckt
    path  = base ++ "/loss.csv"
    path' = base ++ "/loss.html"
    tl    = map float2Double tl'
    vl    = map float2Double vl'
    ep    = map fromIntegral [0 .. length tl - 1]
    csv   = unlines . ("train,valid":)
          $ zipWith (\t v -> show t ++ "," ++ show v) tl' vl'
    cfg   = Plt.defaultConfig {Plt.lineMode = Plt.Lines, Plt.ymode = Plt.Log}

testModel :: PDK.ID -> CKT.ID -> (T.Tensor -> T.Tensor) -> [String] -> [String]
          -> DF.DataFrame T.Tensor -> IO ()
testModel pdk ckt mdl paramsX paramsY df = do
    createDirectoryIfMissing True plotPath
    dat <- DF.sampleIO 1000 False df
    -- dat <- DF.shuffleIO df
    let xs  = DF.values $ DF.lookup paramsX dat
        ys  = DF.values $ DF.lookup paramsY dat
        ys' = mdl xs
    mapM_ (uncurry' (plt plotPath))  $ zip3 paramsY (T.cols ys) (T.cols ys') 
    mapM_ (uncurry  (hst plotPath))  $ zip  paramsY (T.cols ys)
    mapM_ (uncurry  (hst' plotPath)) $ zip  paramsX (T.cols xs)
    dat' <- DF.sampleIO 10 False df
    let x  = DF.values $ DF.lookup paramsX dat'
        y  = DF.values $ DF.lookup paramsY dat'
        y' = DF.values . DF.dropNan . DF.DataFrame paramsY $ mdl x
    print x
    print y
    print y'
    dat'' <- DF.sampleIO 100 False df
    let ex = DF.values $ DF.lookup paramsX dat''
        ey = DF.values $ DF.lookup paramsY dat''
        ey' = DF.values . DF.dropNan . DF.DataFrame paramsY $ mdl ex
        ae = (/ ey) $ T.l1Loss T.ReduceNone ey ey'
        mae = T.meanDim (T.Dim 0) T.RemoveDim T.Float ae
        mape = 100 * mae
        mape' = T.meanAll mape
    print mae
    print mape
    print mape'
  where
    plotPath = "./plots/" ++ show pdk ++ "/" ++ show ckt

hst' :: FilePath -> String -> T.Tensor -> IO ()
hst' path' label y = Plt.plot path $ Plt.histogram [label] [vy] cfg
  where
    vy   = if "_fug" `isSuffixOf` label
              then T.asValue . T.log10 . T.toDType T.Double $ y :: [Double]
              else T.asValue . T.toDType T.Double $ y :: [Double]
    path = path' ++ "/" ++ label ++ "-pde.html"
    cfg  = Plt.defaultConfig { Plt.title' = label
                             , Plt.xlabel = label
                             , Plt.ylabel = "frequency" }

hst :: FilePath -> String -> T.Tensor -> IO ()
hst path' label y = Plt.plot path $ Plt.histogram [label] [vy] cfg
  where
    vy   = vec2lst label y
    path = path' ++ "/" ++ label ++ "-pde.html"
    cfg  = Plt.defaultConfig { Plt.title' = label
                             , Plt.xlabel = label
                             , Plt.ylabel = "frequency" }

plt :: FilePath -> String -> T.Tensor -> T.Tensor -> IO ()
plt path' label y y' = Plt.plot path $ Plt.scatter' [label] [vy] [vy'] cfg
  where
    vy   = vec2lst label y
    vy'  = vec2lst label y'
    path = path' ++ "/" ++ label ++ "-scatter.html"
    cfg  = Plt.defaultConfig { Plt.title'   = label
                             , Plt.xlabel   = "real"
                             , Plt.ylabel   = "pred" 
                             , Plt.lineMode = Plt.Markers
                             , Plt.marker   = Plt.Marker 10.0 Plt.Circle 1.0
                             , Plt.legend   = False 
                             , Plt.width    = 1000
                             , Plt.height   = 1000 }

vec2lst :: String -> T.Tensor -> [Double]
vec2lst "ugbw" = T.asValue . T.log10 . T.toDType T.Double
vec2lst "area" = T.asValue . T.log10 . T.toDType T.Double
vec2lst "sr_r" = T.asValue . T.log10 . T.toDType T.Double
vec2lst "sr_f" = T.asValue . T.log10 . T.toDType T.Double . T.abs
vec2lst _      = T.asValue           . T.toDType T.Double
