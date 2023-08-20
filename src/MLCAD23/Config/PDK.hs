{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DuplicateRecordFields #-}

-- | Configuration File Parsers for Serafin
module MLCAD23.Config.PDK  where

import           GHC.Generics
import           Data.Yaml
import qualified Data.Map                  as M
import           MLCAD23.Util
import qualified Torch                     as T
import qualified Torch.Extensions          as T
import qualified Torch.Functional.Internal as T (powScalar)

-- | Available PDKs
data ID = XH035   -- ^ X-Fab 350nm Process
        | XH018   -- ^ X-Fab 180nm Process
        | XT018   -- ^ X-Fab 180nm Process
        | SKY130  -- ^ SkyWater 130nm Process
        | GPDK180 -- ^ Cadence Generic PDK 180nm Process
        | GPDK090 -- ^ Cadence Generic PDK 90nm Process
        | GPDK045 -- ^ Cadence Generic PDK 45nm Process
    deriving (Eq)

-- | Show Instance of PDK
instance Show ID where
  show XH035   = "xh035"
  show XH018   = "xh018"
  show XT018   = "xt018"
  show SKY130  = "sky130"
  show GPDK180 = "gpdk180"
  show GPDK090 = "gpdk090"
  show GPDK045 = "gpdk045"

-- | Show Instance of PDK
instance Read ID where
  readsPrec _ "xh035"   = [(XH035,   "")]
  readsPrec _ "xh018"   = [(XH018,   "")]
  readsPrec _ "xt018"   = [(XT018,   "")]
  readsPrec _ "sky130"  = [(SKY130,  "")]
  readsPrec _ "gpdk180" = [(GPDK180, "")]
  readsPrec _ "gpdk090" = [(GPDK090, "")]
  readsPrec _ "gpdk045" = [(GPDK045, "")]
  readsPrec _ _         = undefined

-- | Process Corneres
data Corner = TM -- ^ 0, Typical Mean / Nominal (NN)
            | WP -- ^ 1, Worst Power: Fast N, Fast P (FF)
            | WS -- ^ 2, Worst Speed: Slow N, Slow P (SS)
            | WZ -- ^ 3, Worst Zero:  Slow N, Fast P (SF)
            | WO -- ^ 4, Worst One:   Fast N, Slow P (FS)
            | MC -- ^ 5, Statistical Model
            deriving (Eq, Show, Enum, Bounded, Read)

-- instance Show Corner where
--   show TM = "tm"
--   show WP = "wp"
--   show WS = "ws"
--   show WZ = "wz"
--   show WO = "wo"
--   show MC = "mc_g"

-- instance Read Corner where
--   readsPrec _ "tm" = [(TM, "")]
--   readsPrec _ "wp" = [(WP, "")]
--   readsPrec _ "ws" = [(WS, "")]
--   readsPrec _ "wz" = [(WZ, "")]
--   readsPrec _ "wo" = [(WO, "")]
--   readsPrec _ "mc_g" = [(MC, "")]
--   readsPrec _ _    = undefined

showCorner :: ID -> Corner -> String
showCorner XH018   TM = "tm"
showCorner XH018   WP = "wp"
showCorner XH018   WS = "ws"
showCorner XH018   WZ = "wz"
showCorner XH018   WO = "wo"
showCorner XH018   MC = "mc_g"
showCorner XT018   TM = "tm"
showCorner XT018   WP = "wp"
showCorner XT018   WS = "ws"
showCorner XT018   WZ = "wz"
showCorner XT018   WO = "wo"
showCorner XT018   MC = "mc_g"
showCorner GPDK180 TM = "nn"
showCorner GPDK180 WP = "ff"
showCorner GPDK180 WS = "ss"
showCorner GPDK180 WZ = "sf"
showCorner GPDK180 WO = "fs"
showCorner GPDK180 MC = "mc_g"
showCorner GPDK090 TM = "nn"
showCorner GPDK090 WP = "ff"
showCorner GPDK090 WS = "ss"
showCorner GPDK090 WZ = "sf"
showCorner GPDK090 WO = "fs"
showCorner GPDK090 MC = "mc_g"
showCorner GPDK045 TM = "nn"
showCorner GPDK045 WP = "ff"
showCorner GPDK045 WS = "ss"
showCorner GPDK045 WZ = "sf"
showCorner GPDK045 WO = "fs"
showCorner GPDK045 MC = "mc_g"
showCorner _       _  = "tm"

-- | Additinal include Paths
data Include = Include { path    :: !FilePath
                       , section :: !String
                       } deriving (Generic, Show)
instance FromJSON Include
instance ToJSON   Include

-- | Geometric Constraints
data Constraint = Constraint { min' :: !Double
                             , max' :: !Double
                             , grid :: !Double
                             } deriving (Generic, Show)
instance FromJSON Constraint where
  parseJSON (Object v) = do
        min''  <- v .:  "min"
        max''  <- v .:  "max"
        grid'  <- v .:  "grid"
        pure $ Constraint min'' max'' grid'
  parseJSON _ = fail "Expected an Object"
instance ToJSON Constraint where
  toJSON Constraint{..} = object [ "min"  .= min'
                                 , "max"  .= max'
                                 , "grid" .= grid ]

-- | Geometric Constraints imposed by PDK
data Constraints = Constraints { length :: !Constraint
                               , width  :: !Constraint 
                               } deriving (Generic, Show)
instance FromJSON Constraints
instance ToJSON   Constraints

-- | DC Operating Point Parameters
data DCOPConfig = DCOPConfig { prefix     :: !String
                             , suffix     :: !String 
                             , parameters :: ![String]
                             } deriving (Generic, Show)
instance FromJSON DCOPConfig
instance ToJSON DCOPConfig

-- | DC Match Analysis Configuration
data DCMConfig = DCMConfig { prefix     :: !String
                           , suffix     :: !String 
                           , reference  :: !String
                           } deriving (Generic, Show)
instance FromJSON DCMConfig
instance ToJSON DCMConfig

-- | DC Match Analysis Mapping for NMOS and PMOS
data DCMConfig' = DCMConfig' { nmos :: ![DCMConfig] 
                             , pmos :: ![DCMConfig]
                             } deriving (Generic, Show)
instance FromJSON DCMConfig' where
  parseJSON (Object v) = do
        nmos' <- v .:  "NMOS"
        pmos' <- v .:  "PMOS"
        pure $ DCMConfig' nmos' pmos'
  parseJSON _ = fail "Expected an Object"
instance ToJSON DCMConfig' where
  toJSON DCMConfig'{..} = object ["NMOS" .= nmos, "PMOS" .= pmos]

-- | Simulation Analysis references
data AnalysisConfig = AnalysisConfig { dcop    :: !DCOPConfig
                                     , dcmatch :: !DCMConfig'
                                     } deriving (Generic, Show)
instance FromJSON AnalysisConfig
instance ToJSON AnalysisConfig

-- | PDK Configuratoin
data CFG = CFG { include     :: ![Include]
               , testbench   :: !(M.Map String Double)
               , defaults    :: !(M.Map String Double)
               , constraints :: !Constraints
               , devices     :: !AnalysisConfig
               } deriving (Generic, Show)
instance FromJSON CFG
instance ToJSON   CFG

-- | Read A Circuit configuration file
readConfig :: FilePath -> IO CFG
readConfig = decodeFileThrow

-- | Calculate Geometry given Resistance Value R -> [W,L]
resistance :: ID -> Passive
resistance pdk r = T.cat (T.Dim 1) [ w, l ]
  where r' = squareResistance pdk
        w  = T.fullLike' @Float r 540.0e-9
        l  = T.divScalar r' $ r * w

-- | Calculate Resitance Value given Geometry [W,L] -> R
resistance' :: ID -> Passive
resistance' pdk wl = T.reshape [-1,1] . T.mulScalar r $ T.div l w
  where r = squareResistance pdk
        w = T.select 1 0 wl
        l = T.select 1 1 wl

-- | Fake values, see PDK MANUAL
squareResistance :: ID -> Float
squareResistance XT018   = 300.0
squareResistance XH018   = 300.0
squareResistance GPDK180 = 7.5
squareResistance GPDK090 = 0.5
squareResistance GPDK045 = 0.08
squareResistance _       = undefined

-- | Calculate Geometry given Capacitance Value C -> [W,L]
capacitance :: ID -> Passive
capacitance pdk c = T.cat (T.Dim 1) [w, w]
  where
    c' = sheetCapacitance pdk
    w = T.mulScalar @Float 1.0e-6 . T.sqrt . T.divScalar c' $ c

-- | Calculate Capacitance Value given Geometry [W,L] -> C
capacitance' :: ID -> Passive
capacitance' pdk wl = T.reshape [-1,1] . T.mulScalar c
                    . T.prodDim (T.Dim 1) T.KeepDim $ T.mulScalar @Float 1.0e6 wl
  where c = sheetCapacitance pdk

-- | Smallest possible structure in Technology
minLength :: ID -> T.Tensor
minLength XT018   = 180.0e-9
minLength XH018   = 180.0e-9
minLength XH035   = 350.0e-9
minLength GPDK045 = 45.0e-9
minLength GPDK090 = 90.0e-9
minLength GPDK180 = 180.0e-9
minLength SKY130  = 130.0e-9

-- | Fake values, see PDK manual
sheetCapacitance :: ID -> Float
sheetCapacitance XT018   = 1.0e-15
sheetCapacitance XH018   = 1.0e-15
sheetCapacitance GPDK045 = 1.1e-15
sheetCapacitance GPDK090 = 1.0e-15
sheetCapacitance GPDK180 = 1.0e-15
sheetCapacitance _       = undefined

-- | Unit squar for normalizing Area
square :: ID -> T.Tensor
square = flip T.powScalar 2.0 . minLength
