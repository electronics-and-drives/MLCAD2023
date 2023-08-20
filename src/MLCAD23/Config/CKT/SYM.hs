{-# OPTIONS_GHC -Wall -fplugin Debug.Breakpoint #-}

{-# LANGUAGE TypeApplications #-}

module MLCAD23.Config.CKT.SYM where

import           Data.Frame                      ((??))
import qualified Data.Frame                as DF
import qualified Torch                     as T
import qualified Torch.Extensions          as T
import           MLCAD23.Util

-- import           Debug.Breakpoint

electricalParams :: [String]
electricalParams = [ "MNDP11_gmoverid", "MNCM11_gmoverid", "MPCM31_gmoverid", "MNCM41_gmoverid"
                   , "MNDP11_fug", "MNCM11_fug", "MPCM31_fug", "MNCM41_fug"
                   , "MNCM12_id", "MNCM42_id"
                   , "MNDP11_vds", "MNCM11_vds", "MPCM31_vds", "MNCM41_vds" ]

geometricalParams :: [String]
geometricalParams = [ "Ldp1",  "Lcm1",  "Lcm2",  "Lcm3",  "Lcm4"
                    , "Wdp1",  "Wcm1",  "Wcm2",  "Wcm3",  "Wcm4"
                    , "Mdp11", "Mcm11", "Mcm21", "Mcm31", "Mcm41"
                    , "Mdp12", "Mcm12", "Mcm22", "Mcm32", "Mcm42" ]

geometricalParams' :: [String]
geometricalParams' = [ "Ldp1",  "Lcm1",  "Lcm2",  "Lcm3",  "Lcm4"
                     , "Wdp1",  "Wcm1",  "Wcm2",  "Wcm3",  "Wcm4"
                     , "Mcm12", "Mcm22" ]

transform :: Passive -> Passive -> Primitive -> Primitive
          -> DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor
transform _ _ nmos pmos elec = geom
  where
    num    = DF.nRows elec
    -- vdd    = T.full' [num,1] vdd'
    vdd    = elec ?? "vdd"
    vss    = T.zeros' [num,1]
    -- i0     = T.abs $ T.full' [num,1] i0'
    i0     = elec ?? "i0"
    i1'    = T.abs $ elec ?? "MNCM12_id"
    i2'    = T.abs $ elec ?? "MNCM42_id"
    mcm11  = T.ones' [num,1]
    mcm12  = fst . T.maxDim (T.Dim 1) T.KeepDim . T.cat (T.Dim 1)
                 . (:[mcm11]) . T.round' 0 $ i1' / i0
    mcm31  = T.ones' [num,1]
    mcm32  = fst . T.maxDim (T.Dim 1) T.KeepDim . T.cat (T.Dim 1)
                 . (:[mcm11]) . T.round' 0 $ (2.0 * i2') / i1'
    mcm21  = mcm31
    mcm22  = mcm32
    mdp11  = 2.0 * T.ones' [num,1]
    mdp12  = mdp11
    mcm41  = 2.0 * T.ones' [num,1]
    mcm42  = mcm41
    i1     = i0 * mcm12
    i12    = i1 / 2.0
    i2     = i12 * mcm32
    vcm    = T.abs (elec ?? "MNDP11_vds") + T.abs (elec ?? "MPCM31_vds") - vdd
    cm1In  = T.cat (T.Dim 1) [ i0, elec ?? "MNCM11_gmoverid", elec ?? "MNCM11_fug"
                             , elec ?? "MNCM11_vds", vss ]
    cm3In  = T.cat (T.Dim 1) [ i12, elec ?? "MPCM31_gmoverid", elec ?? "MPCM31_fug"
                             , elec ?? "MPCM31_vds", vss ]
    cm4In  = T.cat (T.Dim 1) [ i2, elec ?? "MNCM41_gmoverid", elec ?? "MNCM41_fug"
                             , elec ?? "MNCM41_vds", vss ]
    dp1In  = T.cat (T.Dim 1) [ i12, elec ?? "MNDP11_gmoverid", elec ?? "MNDP11_fug"
                             , elec ?? "MNDP11_vds", vcm ]
    cm1Out = nmos cm1In
    cm3Out = pmos cm3In
    cm4Out = nmos cm4In
    dp1Out = nmos dp1In
    cm1W   = T.select' 1 0 cm1Out / mcm11
    cm3W   = T.select' 1 0 cm3Out / mcm31
    cm4W   = T.select' 1 0 cm4Out / mcm41
    dp1W   = T.select' 1 0 dp1Out / mdp11
    cm1L   = T.select' 1 1 cm1Out
    cm3L   = T.select' 1 1 cm3Out
    cm4L   = T.select' 1 1 cm4Out
    dp1L   = T.select' 1 1 dp1Out
    cm2L   = cm3L
    cm2W   = cm3W
    geom   = DF.DataFrame geometricalParams
           $ T.cat (T.Dim 1) [ dp1L,  cm1L,  cm2L,  cm3L,  cm4L
                             , dp1W,  cm1W,  cm2W,  cm3W,  cm4W
                             , mdp11, mcm11, mcm21, mcm31, mcm41
                             , mdp12, mcm12, mcm22, mcm32, mcm42 ]


-- "MNCM12_id", "MNCM42_id"
branchMultipliers :: [(Int, Int)]
branchMultipliers = [(1,4),  (3,9)]

-- "MNDP11_vds", "MNCM11_vds", "MPCM31_vds", "MNCM41_vds"
nodeVoltageMultipliers :: [Float]
nodeVoltageMultipliers = [0.7, 0.4, -0.4, 0.4]

trainMask :: DF.DataFrame T.Tensor -> T.Tensor
trainMask df = T.logicalAnd (T.lt (df ?? "gm") 0.0) 
             . T.logicalAnd (T.gt (df ?? "pm") 0.0)
             . T.logicalAnd (T.gt (df ?? "ugbw") 10000.0)
             . T.logicalAnd (T.gt (df ?? "sr_r") 10000.0)
             $ T.logicalAnd (T.lt (df ?? "sr_f") (-10000.0))
                            (T.gt (df ?? "a_0") 25.0)

-- | ((Mcm11+Mcm12)*Wcm1*Lcm1)+((Mdp11+Mdp12)*Wdp1*Ldp1)+((Mcm21+Mcm22)*Wcm2*Lcm2)+((Mcm31+Mcm32)*Wcm3*Lcm3)+(Mcm41+Mcm42)*Wcm4*Lcm4
area :: DF.DataFrame T.Tensor -> T.Tensor
area df = ((mcm11 + mcm12) * wcm1 * lcm1)
        + ((mdp11 + mdp12) * wdp1 * ldp1)
        + ((mcm21 + mcm22) * wcm2 * lcm2)
        + ((mcm31 + mcm32) * wcm3 * lcm3)
        + ((mcm41 + mcm42) * wcm4 * lcm4)
  where
    ldp1  = df ?? "Ldp1"
    lcm1  = df ?? "Lcm1"
    lcm2  = df ?? "Lcm2"
    lcm3  = df ?? "Lcm3"
    lcm4  = df ?? "Lcm4"
    wdp1  = df ?? "Wdp1"
    wcm1  = df ?? "Wcm1"
    wcm2  = df ?? "Wcm2"
    wcm3  = df ?? "Wcm3"
    wcm4  = df ?? "Wcm4"
    mdp11 = df ?? "Mdp11"
    mdp12 = df ?? "Mdp12"
    mcm11 = df ?? "Mcm11"
    mcm12 = df ?? "Mcm12"
    mcm21 = df ?? "Mcm21"
    mcm22 = df ?? "Mcm22"
    mcm31 = df ?? "Mcm31"
    mcm32 = df ?? "Mcm32"
    mcm41 = df ?? "Mcm41"
    mcm42 = df ?? "Mcm42"
