{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Extensions to Torch
module Torch.Extensions where

import           GHC.Generics
import           GHC.Float                            (float2Double)
import           Data.List                            (elemIndex)
import           Data.Maybe                           (fromJust, isJust)
import           Data.Complex
import           Data.Ratio
import           System.IO.Unsafe                     (unsafePerformIO)
import qualified Data.Vector.Storable         as V
import qualified Torch                        as T    hiding (linear)
import qualified Torch.Functional.Internal    as T    ( nan_to_num, powScalar', roll
                                                      , linalg_multi_dot, block_diag
                                                      , avg_pool2d, sort, where', cov
                                                      , angle, cartesian_prod, pdist
                                                      , linalg_cholesky, linalg_solve
                                                      , maximum, minimum, special_round
                                                      , meanAll, mse_loss, huber_loss
                                                      , round_t, neg, prodDim, prodAll
                                                      , negative, linear, stdDim 
                                                      , pad_sequence )

import qualified Torch.Optim.CppOptim         as T
import qualified Torch.Internal.Cast          as ATen
import qualified Torch.Internal.Managed.Optim as TM

-- | Get parameters from Optimizer state
getParams :: T.CppOptimizerState option -> IO [T.Parameter]
getParams (T.CppOptimizerState _ opt) = do
    params <- ATen.cast1 TM.getParams opt
    pure $ map (T.IndependentTensor . T.Unsafe) params

-- | Type alias for Padding Kernel
type Kernel  = (Int, Int)

-- | Type alias for Padding Stride
type Stride  = (Int, Int)

-- | Type alias for Padding
type Padding = (Int, Int)

-- | GPU
gpu :: T.Device
gpu = T.Device T.CUDA 1

-- | CPU
cpu :: T.Device
cpu = T.Device T.CPU 0

-- | Flipped version of @withTensorOptions@
withOptions :: T.TensorOptions -> T.Tensor -> T.Tensor
withOptions = flip T.withTensorOptions

-- | Empty Float Tensor on CPU
empty :: T.Tensor
empty = T.asTensor ([] :: [Float])

-- | The inverse of `log10`
pow10 :: T.Tensor -> T.Tensor
pow10 = T.powScalar' 10.0

-- | Diagonal
diag' :: T.Tensor -> T.Tensor
diag' = T.diag (T.Diag 0)

-- | 20 * log |x|
db20 :: T.Tensor -> T.Tensor
db20 = T.mulScalar (20.0 :: Float) .  T.log10 . T.abs

-- | 10 ^ ( x / 20)
db20' :: T.Tensor -> T.Tensor
db20' = T.powScalar' 10.0 . T.divScalar (20.0 :: Float)

-- | Like chainMatmul, but that's deprecated
multiDot :: [T.Tensor] -> T.Tensor
multiDot = T.linalg_multi_dot

-- | Product over Tensor Dimension
prodDim :: T.Dim -> T.KeepDim -> T.Tensor -> T.Tensor
prodDim (T.Dim d) kd t = T.prodDim t d kd' dt'
  where
    kd' = kd == T.KeepDim
    dt' = T.dtype t

-- | Product over all Tensor Dimension
prodAll :: T.Tensor -> T.Tensor
prodAll t = T.prodAll t dt'
  where
    dt' = T.dtype t

-- | Cartesian Product
cartesianProduct :: [T.Tensor] -> T.Tensor
cartesianProduct = T.cartesian_prod

-- | p-norm distance between every pair of row vectors in input
pdist :: Int -> T.Tensor -> T.Tensor
pdist p t = T.pdist t (realToFrac p)

-- | cholesky solve for batchsize 1
choleskySolve' :: T.Tensor -> T.Tensor -> T.Tensor
choleskySolve' x1 x2 = T.squeezeDim 0 $ T.choleskySolve T.Lower x1' x2'
  where
    x1' = T.unsqueeze (T.Dim 0) x1
    x2' = T.unsqueeze (T.Dim 0) x2

-- | Linalg cholesky
cholesky' :: T.Tri -> T.Tensor -> T.Tensor
cholesky' T.Lower = flip T.linalg_cholesky False
cholesky' T.Upper = flip T.linalg_cholesky True

-- | CamelCase
linalgSolve :: T.Tensor -> T.Tensor -> T.Tensor
linalgSolve = T.linalg_solve

-- | Multivaraite Normal Sample
multivariateNormalIO' :: Int -> T.Tensor -> T.Tensor -> IO T.Tensor
multivariateNormalIO' n μ σ = do 
    z <- T.toDType T.Double <$> T.randnIO' [d,n]
    pure . T.transpose2D $ T.matmul l z + T.unsqueeze (T.Dim 1) μ
  where
    d = head $ T.shape μ
    l = cholesky' T.Lower σ

-- | Sample from Gumbel Distribution
gumbelIO :: [Int] -> IO T.Tensor
gumbelIO shape = T.neg . T.log . T.addScalar ε . T.neg . T.log . T.addScalar ε
                    <$> T.randIO' shape
  where
    ε = 1.0e-20 :: Float

-- | Apply softmax relaxation to gumbel sample
gumbelSoftmaxIO :: Float -> T.Tensor -> IO T.Tensor
gumbelSoftmaxIO temp logits = T.softmax (T.Dim (-1)) . T.divScalar temp
                            . T.add logits <$> gumbelIO (T.shape logits)

-- | Not using snake_case
blockDiag :: [T.Tensor] -> T.Tensor
blockDiag = T.block_diag

avgPool2D' :: Kernel -> Stride -> Padding -> Bool -> Bool -> Int -> T.Tensor 
           -> T.Tensor
avgPool2D' k s p m c d t = T.avg_pool2d t k s p m c d 

avgPool2D :: Kernel -> Stride -> Padding -> T.Tensor -> T.Tensor
avgPool2D k s p t = T.avg_pool2d t k s p True True 1 

-- | Because snake_case sucks and this project uses Float instead of Double
nanToNum :: Float -> Float -> Float -> T.Tensor -> T.Tensor
nanToNum nan' posinf' neginf' self = T.nan_to_num self nan posinf neginf
  where
    nan    = float2Double nan'
    posinf = float2Double posinf'
    neginf = float2Double neginf'

-- | Default limits for `nanToNum`
nanToNum' :: T.Tensor -> T.Tensor
nanToNum' self = T.nan_to_num self nan posinf neginf
  where
    nan    = 0.0     :: Double
    posinf = 2.0e32  :: Double
    neginf = -2.0e32 :: Double

-- | Default limits for `nanToNum` (0.0)
nanToNum'' :: T.Tensor -> T.Tensor
nanToNum'' self = T.nan_to_num self nan posinf neginf
  where
    nan    = 0.0 :: Double
    posinf = 0.0 :: Double
    neginf = 0.0 :: Double

-- | Syntactic Sugar for minDim
minDim' :: T.Dim -> T.Tensor -> T.Tensor
minDim' dim = fst . T.minDim dim T.RemoveDim

-- | Syntactic Sugar for maxDim
maxDim' :: T.Dim -> T.Tensor -> T.Tensor
maxDim' dim = fst . T.maxDim dim T.RemoveDim

-- | Elementwise comparison of two tensors of equal size. Returns maximum
max' :: T.Tensor -> T.Tensor -> T.Tensor
max' a b = T.where' (T.gt a b) a b

-- | Elementwise comparison of two tensors of equal size. Returns minimum
min' :: T.Tensor -> T.Tensor -> T.Tensor
min' a b = T.where' (T.lt a b) a b

-- | Mean over all dimensions
meanAll :: T.Tensor -> T.Tensor
meanAll t = T.meanAll t (T.dtype t)

-- | Mean of Nonzero Elements only
meanAllNZ :: T.Tensor -> T.Tensor
meanAllNZ t = meanAll t'
  where
    nz = T.asValue @[[Int]] $ T.nonzero t
    t' = T.cat (T.Dim 0) $ map (T.reshape [1] . foldl (flip (T.select 0)) t) nz

-- | Standard Deviation across given dim
stdDim :: T.Dim -> T.KeepDim -> Bool -> T.Tensor -> T.Tensor
stdDim (T.Dim d) k u t = T.stdDim t d u $ k == T.KeepDim

-- | Round Tensor to number of decimals
round' :: Int -> T.Tensor -> T.Tensor
round' = flip T.special_round

-- | Round 
round :: T.Tensor -> T.Tensor
round = T.round_t

softRound :: Float -> T.Tensor -> T.Tensor
softRound α' x = y
  where
    α = T.asTensor α' 
    m = T.addScalar @Float 0.5 $ T.floor x
    r = x - m
    z = T.tanh (α / 2.0) * 2.0
    y = m + T.tanh (α * r) / z

-- | Round differentiable Discrete
δround :: T.Tensor -> T.Tensor -> T.Tensor
δround ys x = y
  where
    m    = 1.0e-9
    len  = T.toDType T.Float . T.asTensor . head $ T.shape ys
    x'   = x * len
    idx  = max' 0.0 . min' (len - 1) $ T.floor x'
    idx' = T.toDType T.Int64 idx
    y'   = T.indexSelect 0 idx' ys
    b    = y' - (m * idx)
    y    = m * x' + b

-- | Order Direction
data Order = Ascending  -- ^ Ascending order
           | Descending -- ^ Descending order
           deriving (Eq, Show)

-- | Sort Tensor
sort :: T.Dim -> Order -> T.Tensor -> (T.Tensor, T.Tensor)
sort (T.Dim dim) ord input = T.sort input dim (ord == Descending)

-- | Quickly select index
index' :: Int -> T.Tensor -> T.Tensor
index' idx = T.index [idx']
  where
    idx' = T.asTensor' idx $ T.withDType T.Int64 T.defaultOpts

-- | same as `select'` but reshapes to column
select' :: Int -> Int -> T.Tensor -> T.Tensor
select' dim idx = T.reshape [-1,1] . T.select dim idx

select'' :: Int -> Int -> T.Tensor -> T.Tensor
select'' dim idx = T.indexSelect dim idx'
  where
    idx' = T.asTensor' [idx] $ T.withDType T.Int64 T.defaultOpts

-- | convert Tensor to List of 1D vectors (columnwise)
cols :: T.Tensor -> [T.Tensor]
cols = map T.squeezeAll . T.split 1 (T.Dim 1)

-- | convert Tensor to List of 1D vectors (rowwise)
rows :: T.Tensor -> [T.Tensor]
rows = map T.squeezeAll . T.split 1 (T.Dim 0)

-- | Vertically stack a list of tensors
vstack :: [T.Tensor] -> T.Tensor
vstack = T.stack (T.Dim 0)

-- | Horizontally stack a list of tensors
hstack :: [T.Tensor] -> T.Tensor
hstack = T.stack (T.Dim 1)

-- | Same as fullLike, but with default options
fullLike' :: (T.Scalar a) => T.Tensor -> a -> T.Tensor
fullLike' t = T.full' $ T.shape t

-- | Shorthand diagonal matrix elements as vector
diagonal2D :: T.Tensor -> T.Tensor
diagonal2D = T.diagonal (T.Diag 0) (T.Dim 0) (T.Dim 1) 

-- | Uniformly sampled values in range [lo;hi]
uniformIO :: T.Tensor -> T.Tensor -> IO T.Tensor
uniformIO lo hi = do
    r <- T.randLikeIO' lo
    pure $ (r * (hi - lo)) + lo

-- | Backdoored Dropout layer
dropout' :: Double -> Bool -> T.Tensor -> T.Tensor
dropout' p t x = unsafePerformIO $ T.dropout p t x

-- | Covariance Matrix
cov'' :: T.Tensor -> T.Tensor
cov'' x = T.cov x' c' fW aW
  where
    x' = T.transpose2D x
    c' = 1
    fW = T.toDType T.Int64 . T.ones' . take 1 $ T.shape x
    aW = T.ones' . take 1 $ T.shape x

-- | Covariance of 2 variables
cov' ::  T.Tensor -> T.Tensor -> T.Tensor
cov' x y = c' / n'
  where
    n' = T.toDType T.Float $ T.asTensor (head $ T.shape x :: Int)
    x' = T.mean x
    y' = T.mean y
    c' = T.sumAll $ (x - x') * (y - y')

-- | Estimates the covariance matrix of the variables given by the `input`
-- matrix, where rows are the variables and columns are the observations.
cov :: T.Tensor -> T.Tensor
cov x = nanToNum'' c
  where
    n  = head $ T.shape x
    n' = T.toDType T.Float $ T.asTensor (last $ T.shape x :: Int)
    μ  = T.meanDim (T.Dim 1) T.KeepDim T.Float x
    x' = x - μ
    cs = [ T.sumDim (T.Dim 1) T.RemoveDim T.Float 
         $ (x' * T.roll x' y 0) / (n' - 1.0)
         | y <- [1 .. n] ]
    c' = T.zeros' [n,n]
    c  = foldl fillC c' . zip (take n . drop 1 $ cycle [0 .. pred n]) $ cs
    fillC :: T.Tensor -> (Int, T.Tensor) -> T.Tensor
    fillC m (i, v) = m'
      where
        o  = T.withDType T.Int64 T.defaultOpts
        z  = T.asTensor' ([0 .. pred n] :: [Int]) o
        s  = T.roll z i 0
        m' = T.indexPut False [z,s] v m

-- | Estimates the Pearson product-moment correlation coefficient matrix of the
-- variables given by the `input` matrix, where rows are the variables and
-- columns are the observations.
corrcoef :: T.Tensor -> T.Tensor
corrcoef x = nanToNum'' . T.clamp (-1.0) 1.0 $ c / c'
  where
    c   = cov x
    dc  = diagonal2D c
    dc' = T.reshape [-1,1] dc
    c'  = T.sqrt . T.abs $ dc * dc'

-- | Mean Absolute Percentage Error
apeLoss :: T.Reduction -> T.Tensor -> T.Tensor -> T.Tensor
apeLoss r x y = T.mulScalar @Float 100.0 . T.abs . r' . flip T.div y . T.abs $ x - y
  where
    r' = case r of
            T.ReduceNone -> id
            T.ReduceMean -> meanAll
            T.ReduceSum  -> T.sumAll
 
-- | MSE with reduction
l2Loss :: T.Reduction -> T.Tensor -> T.Tensor -> T.Tensor
l2Loss T.ReduceNone x y = T.mse_loss x y 0
l2Loss T.ReduceMean x y = T.mse_loss x y 1
l2Loss T.ReduceSum  x y = T.mse_loss x y 2

-- | RMSE Error with reduction
rmseLoss :: T.Reduction -> T.Tensor -> T.Tensor -> T.Tensor
rmseLoss r x y | r == T.ReduceNone = rmse
               | r == T.ReduceMean = meanAll rmse
               | r == T.ReduceSum  = T.sumAll rmse
               | otherwise = undefined
  where
    rmse = T.sqrt . T.meanDim (T.Dim 0) T.RemoveDim T.Float
         $ l2Loss T.ReduceNone x y

-- | Huber Loss with δ
huberLoss' :: Double -> T.Reduction -> T.Tensor -> T.Tensor -> T.Tensor
huberLoss' δ T.ReduceNone x y = T.huber_loss x y 0 δ
huberLoss' δ T.ReduceMean x y = T.huber_loss x y 1 δ
huberLoss' δ T.ReduceSum  x y = T.huber_loss x y 2 δ

-- | Huber Loss with default δ = 1.0
huberLoss :: T.Reduction -> T.Tensor -> T.Tensor -> T.Tensor
huberLoss = huberLoss' 1.0

-- | Create a boolean mask from a subset of column names
boolMask :: [String] -> [String] -> [Bool]
boolMask sub = map (`elem` sub)

-- | Create a boolean mask Tensor from a subset of column names
boolMask' :: [String] -> [String] -> T.Tensor
boolMask' sub set = T.asTensor' (boolMask sub set) 
                  $ T.withDType T.Bool T.defaultOpts

-- | like @clamp@ but with higher dimensional bounds
clamp' :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
clamp' l u = T.maximum l . T.minimum u

-- | scale input tensor s.t. ∈ [0,1]
scale01 :: T.Tensor -> T.Tensor
scale01 x = scale xMin xMax x
  where
    xMin = fst . T.minDim (T.Dim 0) T.RemoveDim $ x
    xMax = fst . T.maxDim (T.Dim 0) T.RemoveDim $ x

-- | Scale data given a min and max
scale :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
scale xMin xMax x = (x - xMin) / (xMax - xMin)

-- | Un-Scale data from [0,1]
scale' :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
scale' xMin xMax x = (x * (xMax - xMin)) + xMin

-- | Apply log10 to masked data
trafo :: T.Tensor -> T.Tensor -> T.Tensor
trafo b x = y + y' -- T.where' xMask' (T.log10 $ T.abs x) x
  where
    m  = T.toDType T.Float . T.logicalAnd b . T.lt 0.0 $ T.abs x 
    m' = 1.0 - m -- T.toDType T.Float . T.logicalNot . T.logicalAnd b $ T.gt x 0.0
    o  = T.mul m' $ T.onesLike x
    y  = T.log10 . T.abs . T.add o $ x * m
    y' = x * m'

-- | Apply pow10 to masked data
trafo' :: T.Tensor -> T.Tensor -> T.Tensor
trafo' b x = y + y' -- T.where' b (pow10 x) x
  where
    m  = T.toDType T.Float b
    m' = T.toDType T.Float $ T.logicalNot b
    y  = T.mul m . pow10 $ x * m
    y' = x * m'

-- | Convert masked data to dB
td :: T.Tensor -> T.Tensor -> T.Tensor
td m x = T.where' m (db20 x) x

-- | Convert masked data from dB
td' :: T.Tensor -> T.Tensor -> T.Tensor
td' m x = T.where' m (db20' x) x

-- | Probability Density Function of Gaussian/Normal distribution
pdf :: (T.Scalar s, Floating s, Fractional s) => s -> s -> T.Tensor -> T.Tensor
pdf μ σ = T.mulScalar (1 / (σ * sqrt (2 * pi))) . T.exp . T.mulScalar @Float (-0.5)
        . T.pow @Float 2.0 . T.divScalar σ . T.subScalar μ

-- evalModel :: T.ScriptModule -> T.Tensor -> T.Tensor
-- evalModel m x = y
--   where
--     T.IVTensor y = T.forward m [T.IVTensor x]
--
-- loadModule :: FilePath -> IO (T.Tensor -> T.Tensor)
-- loadModule modPath = evalModel <$> T.loadScript T.WithoutRequiredGrad modPath

-- | Save a torch module
saveModule :: T.RuntimeMode -> FilePath -> Int -> (T.Tensor -> T.Tensor) -> IO ()
saveModule mode path num mdl = do
    data' <- T.randIO' [1,num]
    rm <- T.trace name "forward" fun [data']
    T.setRuntimeMode rm mode
    sm <- T.toScriptModule rm
    T.saveScript sm path
  where
    fun       = mapM (pure . mdl)
    name      = "fuacide"
    -- data'     = [T.ones' [1, num]]

-- | Save a torch module in Eval Mode
saveEvalModule :: FilePath -> Int -> (T.Tensor -> T.Tensor) -> IO ()
saveEvalModule = saveModule T.Eval

-- | load a torch module
loadModule :: T.LoadMode -> FilePath -> IO (T.Tensor -> T.Tensor)
loadModule mode path = do
    mdl <- T.loadScript mode path
    let fun x = let T.IVTensor y = T.runMethod1 mdl "forward" $ T.IVTensor x
                  in y
    pure fun

-- | load a torch module without grad
loadEvalModule  :: FilePath -> IO (T.Tensor -> T.Tensor)
loadEvalModule = loadModule T.WithoutRequiredGrad

-- | Trace model with parameter names
traceModelWithNames :: [String] -> [String] -> Int -> (T.Tensor -> T.Tensor) -> IO T.RawModule
traceModelWithNames xs ys n f = do
    rm <- T.trace "fuacide" "forward" fun data'
    T.define rm $ "def inputs(self,x):\n\treturn " ++ show xs ++ "\n"
    T.define rm $ "def outputs(self,x):\n\treturn " ++ show ys ++ "\n"
    pure rm
  where
    fun   = mapM (pure . f)
    data' = [T.ones' [1, n]]

-- | Function to ScriptModule
traceModel :: Int -> (T.Tensor -> T.Tensor) -> IO T.ScriptModule
traceModel n f = T.randnIO' [10,n] >>= T.trace "fuacide" "forward" fun . (: [])
                    >>= T.toScriptModule
  where
    fun         = mapM (pure . f)
    -- data' = [T.ones' [1, n]]

traceModel' :: T.Tensor -> (T.Tensor -> T.Tensor) -> IO T.RawModule
traceModel' data' f = T.trace "fuacide" "forward" fun [data']
  where
    fun   = mapM (pure . f)

-- | Function to RawModule
traceModelIO :: Int -> (T.Tensor -> IO T.Tensor) -> IO T.RawModule
traceModelIO n f = T.trace "fuacide" "forward" (mapM f) data'
  where
    data' = [T.ones' [1, n]]

-- | Trace to Function
unTraceModel' ::  T.ScriptModule -> (T.Tensor -> T.Tensor)
unTraceModel' model' x = y
  where
    T.IVTensor y = T.forward model' [T.IVTensor x]

-- | Trace to Function
unTraceModel :: T.ScriptModule -> (T.Tensor -> T.Tensor)
unTraceModel model' x = y
  where
    T.IVTensor y = T.runMethod1 model' "forward" $ T.IVTensor x

-- | Save model without gradients as script
saveInferenceModel :: FilePath -> T.ScriptModule -> IO ()
saveInferenceModel path model = T.saveScript model path

-- | Load model without gradients from script
loadInferenceModel :: FilePath -> IO T.ScriptModule
loadInferenceModel = T.loadScript T.WithoutRequiredGrad

-- | Load model without gradients from script
loadOptimizationModel :: FilePath -> IO T.ScriptModule
loadOptimizationModel = T.loadScript T.WithRequiredGrad

-- | Load a Pickled Tensor from file
loadTensor :: FilePath -> IO T.Tensor
loadTensor path = do
    T.IVTensor t <- T.pickleLoad path
    pure t

-- | Pickle a Tensor and Save to file
saveTensor :: T.Tensor -> FilePath -> IO ()
saveTensor t path = do
    let t' = T.IVTensor t
    T.pickleSave t' path

-- | List of rows
asRows :: T.Tensor -> [T.Tensor]
asRows = map T.squeezeAll . T.split 1 (T.Dim 0)

-- | List of columns
asCols' :: T.KeepDim -> T.Tensor -> [T.Tensor]
asCols' T.KeepDim   = T.split 1 (T.Dim 1)
asCols' T.RemoveDim = map T.squeezeAll . T.split 1 (T.Dim 1)

-- | List of columns
asCols :: T.Tensor -> [T.Tensor]
asCols = asCols' T.RemoveDim

-- | Convert IVTensor to Tensor
fromIV :: T.IValue  -> T.Tensor
fromIV (T.IVTensor t) = t 
fromIV _ = undefined

-- | Convert Tensor to IVTensor
toIV :: T.Tensor -> T.IValue
toIV = T.IVTensor

-- | Create Integer Index for a subset of column names
intIdx :: [String] -> [String] -> [Int]
intIdx set = fromJust . sequence . filter isJust . map (`elemIndex` set)

-- | Create Integer Index Tensor for a subset of column names
intIdx' :: [String] -> [String] -> T.Tensor
intIdx' set sub = T.asTensor' (intIdx set sub) 
                $ T.withDType T.Int64 T.defaultOpts

-- | Convert Double Vector to Float Tensor
fromRealWave :: V.Vector Double -> T.Tensor
fromRealWave = T.toDType T.Float . T.asTensor . V.toList

-- | Convert Complex Double Vector to Complex Float Tensor
fromImagWave :: V.Vector (Complex Double) -> T.Tensor
fromImagWave = T.toDType T.ComplexFloat . T.asTensor . V.toList

-- | Find Closest Index
findClosestIdx :: T.Tensor -> Float -> T.Tensor
findClosestIdx t x = T.argmin x' 0 False
  where
    x' = T.abs . T.subScalar x $ T.squeezeAll t

-- | Find closest Index as Int
findClosestIdx' :: T.Tensor -> Float -> Int
findClosestIdx' t x = T.asValue $ findClosestIdx t x

-- | Angle of Complex Tensor in Degrees
phase :: T.Tensor -> T.Tensor
phase = T.divScalar (pi :: Float) . T.mulScalar (180.0 :: Float) . T.angle

padStack :: Bool -> Double -> [T.Tensor] -> T.Tensor
padStack batchFirst padValue ts = T.toDType (T.dtype $ head ts)
                                $ T.pad_sequence ts batchFirst padValue

-- | Approximate Integer Ratio
approxRational' :: Float -> T.Tensor -> (T.Tensor, T.Tensor)
approxRational' ε t = (n, d)
  where
    t' = T.asValue t :: [Float]
    r' = map (`approxRational` ε) t'
    n' = map (realToFrac . numerator)   r' :: [Float]
    d' = map (realToFrac . denominator) r' :: [Float]
    n  = T.reshape (T.shape t) $ T.asTensor n'
    d  = T.reshape (T.shape t) $ T.asTensor d'

-- | Linear BNN Layer Specification
data BayesLinearSpec = BayesLinearSpec { in_features     :: !Int
                                       , out_features    :: !Int
                                       , prior_mu        :: !Float
                                       , prior_log_sigma :: !Float
                                       } deriving (Show, Eq)

-- | Linear BNN Layer
data BayesLinear = BayesLinear { weight_mu        :: !T.Parameter
                               , weight_log_sigma :: !T.Parameter
                               , bias_mu          :: !T.Parameter
                               , bias_log_sigma   :: !T.Parameter
                               } deriving (Show, Generic, T.Parameterized)

instance T.Randomizable BayesLinearSpec BayesLinear where
  sample BayesLinearSpec{..} = do
    let stdv = (1.0 /) . T.sqrt . realToFrac $ in_features
    weight_mu' <- scale' (T.negative stdv) stdv
                <$> T.randIO' [out_features, in_features]
    let weight_log_sigma' = fullLike' weight_mu' prior_log_sigma
    bias_mu' <- scale' (T.negative stdv) stdv <$> T.randIO' [out_features]
    let bias_log_sigma' =  fullLike' bias_mu' prior_log_sigma
    BayesLinear <$> T.makeIndependent weight_mu'
                <*> T.makeIndependent weight_log_sigma'
                <*> T.makeIndependent bias_mu'
                <*> T.makeIndependent bias_log_sigma'

bayesLinear' :: BayesLinear -> T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
bayesLinear' BayesLinear{..} input weight_eps bias_eps = T.linear input weight bias
  where
    weight = T.add (T.toDependent weight_mu) $ T.exp (T.toDependent weight_log_sigma) * weight_eps
    bias   = T.add (T.toDependent bias_mu) $ T.exp (T.toDependent bias_log_sigma) * bias_eps

bayesLinear :: BayesLinear -> T.Tensor -> IO T.Tensor
bayesLinear layer@BayesLinear{..} input = do
    weight_eps <- T.randIO' . T.shape $ T.toDependent weight_mu
    bias_eps   <- T.randIO' . T.shape $ T.toDependent bias_mu
    pure $ bayesLinear' layer input weight_eps bias_eps 

instance T.HasForward BayesLinear T.Tensor T.Tensor where
  forward layer input = unsafePerformIO $ bayesLinear layer input
  forwardStoch = bayesLinear
