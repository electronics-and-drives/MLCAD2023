{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DuplicateRecordFields #-}

-- | Utility Functions
module MLCAD23.Util where

import           Data.Bifunctor                      (bimap)
import           Data.Time.Clock                     (getCurrentTime)
import           Data.Time.Format                    (formatTime, defaultTimeLocale)
import           Rando                               (shuffle)
import           Control.Monad.State
import           Data.Complex
import           Data.List
import qualified Data.Map             as M
import qualified Data.Vector.Storable as V
import qualified Torch                as T
import qualified Torch.Extensions     as T

lhsMaxMin :: Int -> Int -> Int -> IO T.Tensor 
lhsMaxMin dims samples num = do
    hc <- replicateM num (lhsRandom dims samples)
    let minDist    = T.stack (T.Dim 0) $ map (T.pdist 2) hc
        hypercubes = T.stack (T.Dim 0) hc
        maxminIdx  = T.asValue . T.argmax (T.Dim 0) T.KeepDim . T.maxDim' (T.Dim 1)
                   $ minDist
    pure $ T.select 0 maxminIdx hypercubes

lhsRandom :: Int -> Int -> IO T.Tensor
lhsRandom dims points = do
    hc' <- T.mulScalar inc . T.transpose2D . T.asTensor
         <$> replicateM dims (shuffle [0 .. (points' - 1)])
    rc' <- T.mulScalar inc <$> T.randIO' (T.shape hc')
    pure $ hc' + rc'
  where
    points' = realToFrac points :: Float
    inc     = 1.0 / points' :: Float

-- | Lift State into StateT
liftState :: forall m s a. (Monad m) => State s a -> StateT s m a
liftState s = do
   state1 <- get
   ( let (result', state') = evalState (do { result'' <- s
                                           ; state'' <- get
                                           ; return (result'', state'')
                                           }) state1
      in do
           put state'
           return result' )

-- | Primitive Device
type Primitive = (T.Tensor -> T.Tensor)

-- | Passive Device
type Passive = (T.Tensor -> T.Tensor)

-- | Flanks
data Flank = Rising     -- ^ Rising Flank
           | Falling    -- ^ Falling Flank
    deriving (Show, Eq, Ord)

-- | Enlist single element
singleton :: a -> [a]
singleton a = [a]

-- | Tuple to List
t2l :: (a,a) -> [a]
t2l (x,y) = [x,y]

-- | Vector with `n` elements filled with given value
fill :: Double -> Int -> V.Vector Double
fill v n = V.fromList (replicate n v)

-- | Vector with `n` zeros
zeros :: Int -> V.Vector Double
zeros = fill 0.0

-- | Vector with `n` zeros
ones :: Int -> V.Vector Double
ones = fill 1.0

-- | Transpose a Map of Lists to a List of Maps (keys MUST align)
transposeMap :: (Ord k) => M.Map k [a] -> [M.Map k a]
transposeMap m = [ M.fromList $ zip ks vs | vs <- transpose $ M.elems m ]
  where
    ks = M.keys m

-- | Transpose a List of Maps to a Map of Lists (keys MUST align)
transposeMap' :: (Ord k) => [M.Map k a] -> M.Map k [a]
transposeMap' ms = M.fromList . zip ks . transpose . map M.elems $ ms
  where
    ks = M.keys $ head ms

-- | Transpose Parameters
transposeParams :: (Ord k, V.Storable a) => M.Map k (V.Vector a) -> [M.Map k a]
transposeParams = transposeMap . M.map V.toList

transposeParams' :: (Ord k, V.Storable a) => [M.Map k a] -> M.Map k (V.Vector a)
transposeParams' = M.map V.fromList . transposeMap' 

-- | Repeats moand action n times
repeatM :: (Monad m) => Int -> m a -> m [a]
repeatM n a = sequence [ a | _ <- [1 .. n] ]

-- | Discard results
repeatM_ :: (Monad m) => Int -> m a -> m ()
repeatM_ n a = do
    _ <- repeatM n a
    pure ()

-- | Apply function to both elements of tuple
both :: (a -> b) -> (a,a) -> (b,b)
both f (x,y) = (f x, f y)

-- | Add an element to a tuple
enTuple :: (a,b) -> c -> (a,b,c)
enTuple (a,b) c = (a,b,c)

-- | Flatten Nutplots
-- flattenNutPlots :: [NutPlot] -> NutPlot
-- flattenNutPlots nps | nutPlotType (head nps) == NutRealPlot = flattenRealPlots nps
--                     | otherwise = flattenComplexPlots nps
-- 
-- -- | For use with `unionsWith` etc.
-- flattenNutPlots' :: NutPlot -> NutPlot -> NutPlot
-- flattenNutPlots' np np' = flattenNutPlots [np, np']

-- | y = 20 · log10 |x|
db20' :: (Floating a) => a -> a
db20' =  (*20.0) .  logBase 10 . abs

-- | y = 20 * log10 |x+j|
db20 :: (RealFloat a) => Complex a -> a
db20 =  (*20.0) .  logBase 10 . magnitude

-- | y = ph(x) · 180 / π
angle :: (RealFloat a) => Complex a -> a
angle = (/pi) . (*180.0) . phase 

-- | Find index close to given element
idxCloseTo :: (Ord a, V.Storable a, Floating a) => a -> V.Vector a -> Int
idxCloseTo val = V.minIndex . V.map (abs . (val -))

-- | Find first index where condiition is met
idxFirstOf :: (Ord a, V.Storable a, Floating a) => Flank -> a -> V.Vector a -> Int
idxFirstOf flank val vec | idx' < (V.length vec - 1) = idx'
                         | otherwise                 = 0
  where
    predicate = if flank == Rising then (>) else (<)
    idx'      = V.length $ V.takeWhile (predicate val) vec

-- | Apply same function to all 4 elements of tuple
forAll4 :: (a -> b) -> (a,a,a,a) -> (b,b,b,b)
forAll4 f (a, b, c, d) = (f a, f b, f c, f d)

-- | uncurry 3
uncurry' :: (a -> b -> c -> d) -> (a,b,c) -> d
uncurry' f (a,b,c) = f a b c

-- | Current Timestamp as formatted string
currentTimeStamp :: String -> IO String
currentTimeStamp format = formatTime defaultTimeLocale format <$> getCurrentTime

-- | Current Timestamp with default formatting: "%Y%m%d-%H%M%S"
currentTimeStamp' :: IO String
currentTimeStamp' = currentTimeStamp "%Y%m%d-%H%M%S"

-- | Shoelace formula for calculating area of polygon
shoelace :: [(Float, Float)] -> Float
shoelace = (/ 2) . abs . uncurry (-) . foldr calcSums (0,0) . (<*>) zip (tail. cycle)
  where
    calcSums ((x,y), (a,b)) = bimap (x * b +) (a * y +)
