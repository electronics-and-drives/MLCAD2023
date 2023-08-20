{-# OPTIONS_GHC -Wall -fplugin Debug.Breakpoint #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Multi-Objective Optimization
module MLCAD23.Opt.Soo where

-- import           Debug.Breakpoint
import           Control.Monad
import           System.Clock
import           GHC.Float
import qualified Data.Frame                          as DF
import           Data.Function                              (on)
import           Data.List                                  (minimumBy, partition, isSuffixOf)
import           Moo.GeneticAlgorithm.Continuous
import           Moo.GeneticAlgorithm.Constraints
import qualified Torch                               as T
import qualified Torch.Extensions                    as T
import           MLCAD23.Opt
import qualified MLCAD23.Config.CKT                  as CKT
import qualified MLCAD23.Config.PDK                  as PDK

constraint' :: Double -> Double -> Int -> Constraint Double Double
constraint' min' max' idx' = min' .<= (!! idx') <=. max'

-- | Boundaries
constraints' :: CKT.ID -> [Constraint Double Double]
constraints' ckt = zipWith3 constraint' lb ub [0 .. ]
  where
    lb = replicate (nx ckt) 0
    ub = replicate (nx ckt) 1
    nx CKT.SYM = 10
    nx CKT.MIL = 10
    nx CKT.RFA = 16

logIO' :: (Show a) => Int -> Population a -> IO ()
logIO' i p = do
    putStrLn $ "Iteration " ++ show i ++ ": Objective Value = " ++ show avg 
    print . head . bestFirst Minimizing $ p
  where
    objs  = map takeObjectiveValue p
    avg   = sum objs / realToFrac (length objs)

logIO :: (Show a) => Int -> IOHook a
logIO n = DoEvery n logIO'

-- | Objective Function
objective' :: PDK.ID -> CKT.ID -> (T.Tensor -> T.Tensor) -> [Float]
          -> ([Double] -> Double)
objective' pdk ckt fun spec = T.asValue @Double . T.toDType T.Double . T.sumAll
                            . cost . fun . trafoX pdk ckt
                            . T.clamp 0.0 1.0 . T.toDType T.Float
                            . T.asTensor
  where
    spec' = T.toDType T.Float $ T.asTensor spec
    cost  = losses spec'

run' :: PDK.ID -> CKT.ID -> IO Result
run' pdk ckt = do
    model <- T.unTraceModel <$> T.loadInferenceModel tracePath
    let fun        = filter' . model
        objective  = objective' pdk ckt fun spec
        step       = withFinalDeathPenalty constraints 
                   $ nextGeneration Minimizing objective selection
                                    eliteSize crossover mutation
        initialize = getRandomGenomes popSize
                   $ replicate dimX (0.0 :: Double, 1.0 :: Double)
    tic <- getTime Realtime
    !population <- runIO initialize $ loopIO [logIO 1] condition step
    toc <- getTime Realtime
    let !rt = (*1.0e-9) . realToFrac . toNanoSecs $ diffTimeSpec toc tic :: Float
    putStrLn $ "Took " ++ show rt ++ "s"

    let x' = takeGenome $ minimumBy (compare `on` takeObjectiveValue) population
        x  = trafoX pdk ckt . T.clamp 0.0 1.0 . T.toDType T.Float $ T.asTensor x'
        y = model x
        l = double2Float $ objective x'
        z = T.sliceDim 0 len (len + len') 1 y
        dfX = DF.union (DF.DataFrame dps $ T.reshape [1,-1] x)
                       (DF.DataFrame vds $ T.reshape [1,-1] z)
        dfY = DF.DataFrame pps . T.reshape [1,-1] $ T.sliceDim 0 0 len 1 y
        df = DF.union dfX dfY

    df' <- validateResult pdk ckt df

    pure $ Result df df' l rt 10
  where
    spec        = specification pdk ckt
    -- tracePath   = "./models/" ++ show ckt ++ "-" ++ show pdk ++ "-trace-" ++ activ ++ ".pt"
    tracePath   = "./models/" ++ show ckt ++ "-" ++ show pdk ++ "-stat-trace.pt"
    filter'     = T.indexSelect' 0 [0 .. 10] . T.squeezeAll -- [ 0, 2, 3, 7, 9 ]
    popSize     = 100
    eliteSize   = 10
    tolerance   = IfObjective $ (< (-1.0e-2)) . minimum
    violation   = degreeOfViolation 10.0 1.0
    -- fitness     = distance2 `on` takeGenome
    selection'  = tournamentSelect Minimizing 25 (popSize - eliteSize)
    selection   = withConstraints constraints violation Minimizing selection'
    crossover   = unimodalCrossoverRP
    mutation    = gaussianMutate 0.1 0.05
    constraints = constraints' ckt
    generations = Generations 20
    condition   = generations `Or` tolerance
    (vds, dps)  = partition (isSuffixOf "_vds") $ CKT.electricalParams ckt
    pps         = [ "a_0", "ugbw", "pm", "gm", "sr_r", "psrr_p", "psrr_n"
                  , "cmrr", "sr_f", "voff_stat", "idd" ]
    dimX        = length dps
    len         = length pps
    len'        = length vds

run :: Int -> PDK.ID -> CKT.ID -> IO ()
run num pdk ckt = replicateM num (run' pdk ckt) >>= evalResults pdk ckt "ga"
