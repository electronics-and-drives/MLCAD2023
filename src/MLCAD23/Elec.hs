{-# OPTIONS_GHC -Wall -fplugin Debug.Breakpoint #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DuplicateRecordFields #-}

-- | Functional Analog Circuit Design
module MLCAD23.Elec where

import           GHC.Float                        (double2Float)
import qualified Torch                     as T
import qualified Torch.Extensions          as T

import           Control.Monad                    ((<$!>))
import           System.Clock
import qualified Data.Frame                as DF
import           Data.List                        ((\\), transpose)
import           Data.List.Split                  (chunksOf)
import qualified Data.Map                  as M
import           Data.Maybe
import qualified Data.Vector.Storable      as V

import qualified MLCAD23.Config.PDK        as PDK
import qualified MLCAD23.Config.CKT        as CKT
import           MLCAD23.Internal
import           MLCAD23.Util
import qualified MLCAD23.Geom              as G

-- import           Debug.Breakpoint

-- | Single Ended Operational Amplifier
data OpAmp = OpAmp { cktId      :: !CKT.ID
                   , pdkId      :: !PDK.ID
                   , elecParams :: ![String]
                   , geomParams :: ![String]
                   , opAmps     :: !G.OpAmps
                   , nmos       :: Primitive
                   , pmos       :: Primitive 
                   , res        :: Passive
                   , cap        :: Passive }

-- | Construct a number of same OpAmps
mkOpAmp :: Int -> CKT.ID -> PDK.ID -> PDK.Corner -> IO OpAmp
mkOpAmp num cktId' pdkId' corner' = do
    opAmps' <- G.mkOpAmps num cktId' pdkId' corner'
    nmos'   <- T.loadEvalModule nmosPath
    pmos'   <- T.loadEvalModule pmosPath
    pure $ OpAmp { cktId      = cktId'
                 , pdkId      = pdkId'
                 , elecParams = CKT.electricalParams cktId'
                 , geomParams = CKT.geometricalParams cktId'
                 , opAmps     = opAmps'
                 , nmos       = nmos'
                 , pmos       = pmos' 
                 , res        = res'
                 , cap        = cap' }
  where
    nmosPath  = circusHome ++ "/pdk/" ++ show pdkId' ++ "/nmos.pt"
    pmosPath  = circusHome ++ "/pdk/" ++ show pdkId' ++ "/pmos.pt"
    res' = PDK.resistance  pdkId'
    cap' = PDK.capacitance pdkId'

numWorkers :: OpAmp -> Int
numWorkers OpAmp{..} = length opAmps

-- | Set list of params to list of values for all opamps
setParameters :: OpAmp -> DF.DataFrame T.Tensor -> IO OpAmp
setParameters op@OpAmp{..} parameters = do
    G.setParams opAmps params
    pure $ op {opAmps = opAmps'}
  where
    params  = DF.asParameters parameters
    params' = M.map V.toList params
    opAmps' = zipWith upd opAmps $ transposeMap params'
    -- rep p v = (p, v')
    --   where
    --     v' = replicate (length opAmps) $ float2Double v
    upd o p = o {G.parameters = parameters'}
      where
        parameters' = M.union p $ G.parameters o

randomBehaviour :: OpAmp -> Int -> Bool -> IO (DF.DataFrame T.Tensor)
randomBehaviour (OpAmp ckt _ _ _ _ _ _ _ _) = CKT.randomBehaviour' ckt

transform :: OpAmp -> DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor
transform OpAmp{..} df = CKT.transform cktId res cap nmos pmos df'
  where
    cols   = ["vdd", "i0", "cl", "temp"]
    defs   = [2.5, 3.0e-6, 5.0e-12]
    def    = DF.DataFrame cols . T.cat (T.Dim 1) $ zipWith lu cols defs
    cols'  = cols \\ DF.columns df
    df'    = if null cols' then df else DF.union df (DF.lookup cols' def)
    lu :: String -> Double -> T.Tensor
    lu p d = T.reshape [-1,1] . T.asTensor . replicate (DF.nRows df)
           . double2Float . fromMaybe d . M.lookup p . G.parameters $ head opAmps

simulate' :: OpAmp -> DF.DataFrame T.Tensor -> IO (DF.DataFrame T.Tensor)
simulate' op@OpAmp{..} elec = do
    mapM_ sim geom
    df <- DF.rowTake (DF.nRows elec) . DF.concat . map DF.fromParameters
            <$> G.readResults opAmps geomT
    pure $ DF.insert ["area"] area df
  where
    tb    = DF.lookup (DF.columns elec \\ elecParams) elec
    sim !g = do
        tic <- getTime Realtime
        G.simulate opAmps g
        toc <- getTime Realtime
        let !td = (*1.0e-9) . realToFrac . toNanoSecs $ diffTimeSpec toc tic :: Float
        putStrLn $ "TOOK: " ++ show td ++ "s"
    !geom' = DF.union tb $ transform op elec
    !geom = map DF.asParameters . DF.chunksOf (numWorkers op) $ geom'
    geomT = ( map (M.map V.fromList . M.fromList
            . zip (M.keys (head geom)) . chunksOf (length geom))
            . transpose . foldl1 (++) . transpose)
                (map (M.elems . M.map V.toList) geom)
    !area = T.reshape [-1, 1] $ CKT.area' cktId geom'

simulate :: OpAmp -> DF.DataFrame T.Tensor -> IO (DF.DataFrame T.Tensor)
simulate op@OpAmp{..} elec = mapM_ sim geom >> pure geom'
  where
    tb    = DF.lookup (DF.columns elec \\ elecParams) elec
    sim !g = do
        tic <- getTime Realtime
        G.simulate opAmps g
        toc <- getTime Realtime
        let !td = (*1.0e-9) . realToFrac . toNanoSecs $ diffTimeSpec toc tic :: Float
        putStrLn $ "TOOK: " ++ show td ++ "s"
    !geom' = DF.union tb $ transform op elec
    !geom  = map DF.asParameters $ DF.chunksOf (numWorkers op) geom'

results :: OpAmp -> DF.DataFrame T.Tensor -> IO (DF.DataFrame T.Tensor)
results op@OpAmp{..} geom' = do
    !perf <- load geom
    let !area = T.reshape [-1, 1] $ CKT.area' cktId perf
    pure $ DF.insert ["area"] area perf
  where
    load !g = do
        tic <- getTime Realtime
        !p  <- map DF.fromParameters <$!> G.readResults opAmps (map DF.asParameters g)
        toc <- getTime Realtime
        let !td = (*1.0e-9) . realToFrac . toNanoSecs $ diffTimeSpec toc tic :: Float
        putStrLn $ "TOOK: " ++ show td ++ "s"
        pure $ DF.concat p -- . DF.concat $ zipWith DF.union g p
    !geom  = map (DF.DataFrame (DF.columns geom') . T.squeezeAll) . T.split 1 (T.Dim 0) . T.transpose (T.Dim 0) (T.Dim 1)
           . T.padStack True 0.0 . T.split (numWorkers op) (T.Dim 0) $ DF.values geom'

close :: OpAmp -> IO ()
close = G.close . opAmps
