{-# OPTIONS_GHC -Wall -fplugin Debug.Breakpoint #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Design space Visualization
module MLCAD23.Viz.Opt where

-- import           Debug.Breakpoint
import           Data.List                            (minimumBy)
import qualified Torch                         as T
import qualified Torch.Extensions              as T
-- import           Data.Frame                           ((??))
import qualified Data.Frame                    as DF

import           Data.Function                        (on)
import           Moo.GeneticAlgorithm.Continuous
import           Moo.GeneticAlgorithm.Constraints

-- | Returns sorted index Tensor
sortIndex :: T.Tensor -> T.Tensor
sortIndex = snd . T.sort (T.Dim 0) T.Ascending

-- | Element-Wise Absolute Difference between two tensors
absDiff :: T.Tensor -> T.Tensor -> T.Tensor
absDiff l r = T.abs $ T.sub l r

logIO' :: [String] -> Int -> Population Double -> IO ()
logIO' ps i p = do
    putStrLn $ "Iteration " ++ show i ++ ": Objective Value = " ++ show avg 
    print cols
  where
    objs  = map takeObjectiveValue p
    avg   = sum objs / realToFrac (length objs)
    cols  = map (ps !!) . T.asValue @[Int] . sortIndex . T.asTensor @[Double]
          . fst . head . bestFirst Minimizing $ p

logIO :: [String] -> Int -> IOHook Double
logIO ps n = DoEvery n (logIO' ps)

-- | Cost Function
cost :: T.Tensor -> [Double] -> Double
cost db x      = y
  where
    colIndex   = sortIndex $ T.asTensor x
    cols       = T.asCols . T.indexSelect 1 colIndex $ db
    colIndices = map sortIndex cols
    colLosses  = T.toDType T.Float . T.stack (T.Dim 1)
               $ zipWith absDiff (take (length colIndices - 1) colIndices)
                                 (drop 1 (cycle colIndices))
    -- loss       = T.meanAll $ T.sumDim (T.Dim 1) T.RemoveDim T.Float colLosses
    -- loss       = T.sumAll $ T.meanDim (T.Dim 0) T.RemoveDim T.Float colLosses
    loss       = T.sumAll colLosses
    y          = T.asValue $ T.toDType T.Double loss

optimizeOrder :: DF.DataFrame T.Tensor -> IO [String]
optimizeOrder DF.DataFrame{..} = do
    !population <- runIO initialize $ loopIO [logIO columns 10] condition step
    let x'            = takeGenome $ minimumBy (compare `on` takeObjectiveValue) population
        sortedColumns = map (columns !!) . T.asValue @[Int] . sortIndex $ T.asTensor x'
        
    pure sortedColumns
  where
    numCols     = length columns
    popSize     = 200
    eliteSize   = 20
    tolerance   = IfObjective $ (< (-1.0e-6)) . minimum
    violation   = degreeOfViolation 1.0 0.0
    -- fitness     = distance2 `on` takeGenome
    selection'  = tournamentSelect Minimizing 50 (popSize - eliteSize)
    selection   = withConstraints constraints violation Minimizing selection'
    crossover   = unimodalCrossoverRP
    mutation    = gaussianMutate 0.1 0.25
    constraints = zipWith3 constraint' [ 0 .. numCols - 1] (repeat 0.0) (repeat 1.0)
    generations = Generations 100
    condition   = generations `Or` tolerance
    objective   = cost values
    step        = withFinalDeathPenalty constraints 
                $ nextGeneration Minimizing objective selection
                                 eliteSize crossover mutation
    initialize  = getRandomGenomes popSize
                $ replicate numCols (0.0 :: Double, 1.0 :: Double)

constraint' :: Int -> Double -> Double -> Constraint Double Double
constraint' idx' min' max' = min' .<= (!! idx') <=. max'
