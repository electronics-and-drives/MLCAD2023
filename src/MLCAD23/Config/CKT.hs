{-# OPTIONS_GHC -Wall -fplugin Debug.Breakpoint #-}

{-# LANGUAGE StrictData #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DuplicateRecordFields #-}

module MLCAD23.Config.CKT where

import           GHC.Generics
import           Data.Yaml
import qualified Data.Map                  as M
import           Data.List                       (isSuffixOf)
import           Data.Frame                      ((??))
import qualified Data.Frame                as DF
import qualified Torch                     as T
import qualified Torch.Extensions          as T
import qualified Torch.Functional.Internal as T  (where')

import           MLCAD23.Util
import qualified MLCAD23.Config.CKT.MIL as MIL
import qualified MLCAD23.Config.CKT.SYM as SYM
import qualified MLCAD23.Config.CKT.RFA as RFA

-- import           Debug.Breakpoint

-- class Transformable c where
--   elecParams :: c -> [String]
--   geomParams :: c -> [String]
--   transform  :: c -> 

geometricalParams :: ID -> [String]
geometricalParams MIL = MIL.geometricalParams
geometricalParams SYM = SYM.geometricalParams
geometricalParams RFA = RFA.geometricalParams

geometricalParams' :: ID -> [String]
geometricalParams' SYM = SYM.geometricalParams'
geometricalParams' _   = undefined

electricalParams :: ID -> [String]
electricalParams MIL = MIL.electricalParams
electricalParams SYM = SYM.electricalParams
electricalParams RFA = RFA.electricalParams

transform :: ID -> Passive -> Passive -> Primitive -> Primitive
          -> DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor
transform MIL = MIL.transform
transform SYM = SYM.transform
transform RFA = RFA.transform

branchMultipliers :: ID -> [(Int, Int)]
branchMultipliers SYM = SYM.branchMultipliers
branchMultipliers MIL = MIL.branchMultipliers
branchMultipliers RFA = RFA.branchMultipliers

nodeVoltageMultipliers :: ID -> [Float]
nodeVoltageMultipliers SYM = SYM.nodeVoltageMultipliers
nodeVoltageMultipliers MIL = MIL.nodeVoltageMultipliers
nodeVoltageMultipliers RFA = RFA.nodeVoltageMultipliers

trainMask :: ID -> DF.DataFrame T.Tensor -> T.Tensor
trainMask MIL = MIL.trainMask
trainMask SYM = SYM.trainMask
trainMask RFA = RFA.trainMask

area' :: ID -> DF.DataFrame T.Tensor -> T.Tensor
area' SYM = SYM.area
area' MIL = MIL.area
area' RFA = RFA.area

randomNodeVoltages :: ID -> T.Tensor -> IO (DF.DataFrame T.Tensor)
randomNodeVoltages id' vdd = DF.DataFrame ids . T.cat (T.Dim 1) <$> mapM rng nvs
  where
    ids     = filter (isSuffixOf "_vds") (electricalParams id')
    nvs     = nodeVoltageMultipliers id'
    rng     = flip T.normalScalarIO 0.0666 . flip T.mulScalar vdd

randomBranchCurrents :: ID -> T.Tensor -> IO (DF.DataFrame T.Tensor)
randomBranchCurrents id' i0 = DF.DataFrame ids . T.cat (T.Dim 1)
                                <$> mapM (uncurry rng) bms
  where
    shape   = T.shape i0
    ids     = filter (isSuffixOf "_id") (electricalParams id')
    bms     = branchMultipliers id'
    rng l u = T.mul i0 . T.toDType T.Float <$> T.randintIO' l u shape

randomBehaviour' :: ID -> Int -> Bool -> IO (DF.DataFrame T.Tensor)
randomBehaviour' id' num pvt = do
    pvDF <- if pvt then DF.DataFrame pvts . T.scale' pvtMin pvtMax <$> lhsMaxMin dims' num 5
                   else pure . DF.DataFrame pvts $ T.repeat [num,1] pvtDef
    idDF <- randomBranchCurrents id' $ pvDF ?? "i0"
    vdDF <- randomNodeVoltages id' $ pvDF ?? "vdd"
    opDF <- DF.DataFrame cols . (\x -> T.where' fugMask (T.pow10 x) x)
                              . T.scale' min' max' <$> lhsMaxMin dims num 5

    pure . DF.lookup (cols' ++ pvts)
         . DF.union pvDF . DF.union vdDF $ DF.union idDF opDF 
  where
    cols'   = electricalParams id'
    gmids   = filter (isSuffixOf "_gmoverid") cols'
    fugs    = filter (isSuffixOf "_fug")      cols'
    cols    = gmids ++ fugs
    gmidNum = length gmids
    fugNum  = length fugs
    dims    = length cols
    gmidMax = T.full' [gmidNum] (20.0 :: Float)
    gmidMin = T.full' [gmidNum] (5.0 :: Float)
    fugMax  = T.full' [fugNum] (10.0 :: Float)
    fugMin  = T.full' [fugNum] (6.0 :: Float)
    max'    = T.cat (T.Dim 0) [gmidMax, fugMax]
    min'    = T.cat (T.Dim 0) [gmidMin, fugMin]
    fugMask = T.boolMask' fugs cols
    pvts    = ["vdd", "i0", "cl", "temp"]
    dims'   = length pvts
    pvtDef  = T.asTensor @[Float] [2.5, 3.0e-6, 5.0e-12 , 27.0]
    pvtMin  = T.asTensor @[Float] [1.8, 3.0e-6, 5.0e-12 , -40.0]
    pvtMax  = T.asTensor @[Float] [3.5, 6.0e-6, 15.0e-12, 150.0]

randomBehaviour :: ID -> Int -> IO (DF.DataFrame T.Tensor)
randomBehaviour id' num = randomBehaviour' id' num False

-- | Available Circuits
data ID = MIL -- ^ miller operational amplifier
        | SYM -- ^ symmetrical operational amplifier
        | RFA -- ^ rail-to rail folded-cascode with wide-swing current mirror
    deriving (Eq)

-- | Show instance of Circuit ID
instance Show ID where
  show MIL = "mil"
  show SYM = "sym"
  show RFA = "rfa"

-- | Read instance of Circuit ID
instance Read ID where
  readsPrec _ "mil" = [(MIL,  "")]
  readsPrec _ "sym" = [(SYM,  "")]
  readsPrec _ "rfa" = [(RFA,  "")]
  readsPrec _ _     = undefined

-- | Available Device Types
data DeviceType = NMOS -- ^ NMOS FET
                | PMOS -- ^ PMOS FET
                | RES  -- ^ RESISTOR
                | CAP  -- ^ CAPACITOR
  deriving (Generic, Show, Eq)
instance FromJSON DeviceType
instance ToJSON DeviceType

-- | Devices in Circuit
data Device = Device { id'   :: !String
                     , type' :: !DeviceType  
                     } deriving (Generic, Show)
instance FromJSON Device where
  parseJSON (Object v) = do
        id''   <- v .:  "id"
        type'' <- v .:  "type"
        pure $ Device id'' type''
  parseJSON _ = fail "Expected an Object"
instance ToJSON Device where
  toJSON Device{..} = object ["id" .= id', "type" .= type']

-- | Configuration Parameters
data ParamConfig = ParamConfig { testbench   :: !(M.Map String Double)
                               , geometrical :: !(M.Map String Double)
                               , area        :: !String
                               } deriving (Generic, Show)
instance FromJSON ParamConfig
instance ToJSON ParamConfig

-- | Main Node in Circuit configuration yaml
data CFG = CFG { devices    :: ![Device]
               , parameters :: !ParamConfig
               } deriving (Generic, Show)
instance FromJSON CFG
instance ToJSON CFG

-- | Read A Circuit configuration file
readConfig :: FilePath -> IO CFG
readConfig = decodeFileThrow
