{-# OPTIONS_GHC -Wall -fplugin Debug.Breakpoint #-}

{-# LANGUAGE TypeApplications #-}

module MLCAD23.Config.CKT.MIL where

import           Data.Frame                      ((??))
import qualified Data.Frame                as DF
import qualified Torch                     as T
import qualified Torch.Extensions          as T
import           MLCAD23.Util

-- import           Debug.Breakpoint

electricalParams :: [String]
electricalParams = [ "MNDP11_gmoverid", "MNCM11_gmoverid", "MPCM21_gmoverid", "MPCS11_gmoverid"
                   , "MNDP11_fug", "MNCM11_fug", "MPCM21_fug", "MPCS11_fug"
                   , "MNCM12_id", "MNCM13_id"
                   , "MNDP11_vds", "MNCM11_vds", "MPCM21_vds", "MPCS11_vds" ]

geometricalParams :: [String]
geometricalParams = [ "Ldp1",  "Lcm1",  "Lcm2",  "Lcs1",  "Lcap"  ,  "Lres"
                    , "Wdp1",  "Wcm1",  "Wcm2",  "Wcs1",  "Wcap"  ,  "Wres"
                    , "Mdp11", "Mcm11", "Mcm21", "Mcs11", "Mcap1" , "Mres1"
                    , "Mdp12", "Mcm12", "Mcm22"
                             , "Mcm13" ]

transform :: Passive -> Passive -> Primitive -> Primitive
          -> DF.DataFrame T.Tensor -> DF.DataFrame T.Tensor
transform res cap nmos pmos elec = geom
  where
    num    = DF.nRows elec
    -- vdd    = T.full' [num,1] vdd'
    vdd    = elec ?? "vdd"
    vss    = T.zeros' [num,1]
    -- i0     = T.abs $ T.full' [num,1] i0'
    i0     = elec ?? "i0"
    -- cl     = T.abs $ T.full' @Float [num,1] 15.0e-12
    cl     = elec ?? "cl"
    i1'    = T.abs $ elec ?? "MNCM12_id"
    i2'    = T.abs $ elec ?? "MNCM13_id"
    mcap1  = T.ones' [num,1]
    mres1  = T.ones' [num,1]
    mcm11  = T.ones' [num,1]
    mcm12  = fst . T.maxDim (T.Dim 1) T.KeepDim . T.cat (T.Dim 1)
                 . (:[mcm11]) . T.round' 0 $ i1' / i0
    mcm13  = fst . T.maxDim (T.Dim 1) T.KeepDim . T.cat (T.Dim 1)
                 . (:[mcm11]) . T.round' 0 $ i2' / i0 
    mcs11  = mcm13
    mdp11  = 2.0 * T.ones' [num,1]
    mdp12  = mdp11
    mcm21  = 2.0 * T.ones' [num,1]
    mcm22  = mcm21
    i1     = i0 * mcm12
    i2     = i0 * mcm13
    i12    = i1 / 2.0
    vcm    = T.abs (elec ?? "MNDP11_vds") + T.abs (elec ?? "MPCM21_vds") - vdd
    cm1In  = T.cat (T.Dim 1) [ i0, elec ?? "MNCM11_gmoverid", elec ?? "MNCM11_fug"
                             , elec ?? "MNCM11_vds", vss ]
    cm2In  = T.cat (T.Dim 1) [ i12, elec ?? "MPCM21_gmoverid", elec ?? "MPCM21_fug"
                             , elec ?? "MPCM21_vds", vss ]
    cs1In  = T.cat (T.Dim 1) [ i2, elec ?? "MPCS11_gmoverid", elec ?? "MPCS11_fug"
                             , elec ?? "MPCS11_vds", vss ]
    dp1In  = T.cat (T.Dim 1) [ i12, elec ?? "MNDP11_gmoverid", elec ?? "MNDP11_fug"
                             , elec ?? "MNDP11_vds", vcm ]
    cp1In  = T.divScalar @Float 3.25e6 $ elec ?? "MNCM12_id"
    rs1In  = (1.0 / ((elec ?? "MPCS11_gmoverid") * i2)) * ((cp1In + cl) / cp1In)
    cm1Out = nmos cm1In
    cm2Out = pmos cm2In
    cs1Out = pmos cs1In
    dp1Out = nmos dp1In
    cp1Out = cap cp1In
    rs1Out = res rs1In
    cm1W   = T.select' 1 0 cm1Out / mcm11
    cm2W   = T.select' 1 0 cm2Out / mcm21
    cs1W   = T.select' 1 0 cs1Out / mcs11
    dp1W   = T.select' 1 0 dp1Out / mdp11
    capW   = T.select' 1 0 cp1Out / mcap1
    resW   = T.select' 1 0 rs1Out / mres1
    cm1L   = T.select' 1 1 cm1Out
    cm2L   = T.select' 1 1 cm2Out
    cs1L   = T.select' 1 1 cs1Out
    dp1L   = T.select' 1 1 dp1Out
    capL   = T.select' 1 1 cp1Out
    resL   = T.select' 1 1 rs1Out
    geom   = DF.DataFrame geometricalParams
           $ T.cat (T.Dim 1) [ dp1L,  cm1L,  cm2L,  cs1L,  capL ,  resL
                             , dp1W,  cm1W,  cm2W,  cs1W,  capW ,  resW
                             , mdp11, mcm11, mcm21, mcs11, mcap1, mres1
                             , mdp12, mcm12, mcm22
                                    , mcm13 ]

-- "MNCM12_id", "MNCM13_id"
branchMultipliers :: [(Int, Int)]
branchMultipliers = [(1,9), (10,41)]

-- "MNDP11_vds", "MNCM11_vds", "MPCM21_vds", "MPCS11_vds"
nodeVoltageMultipliers :: [Float]
nodeVoltageMultipliers = [0.75, 0.4, -0.4, -0.75]

trainMask :: DF.DataFrame T.Tensor -> T.Tensor
trainMask df = T.logicalAnd (T.gt (df ?? "sr_r") 0.0)
             . T.logicalAnd (T.lt (df ?? "sr_f") 0.0)
             . T.logicalAnd (T.gt (df ?? "ugbw") 0.0)
             . T.logicalAnd (T.eq (df ?? "MNCM12_region") 2)
             . T.logicalAnd (T.eq (df ?? "MNCM13_region") 2)
             . T.logicalAnd (T.eq (df ?? "MNDP11_region") 2)
             $ T.logicalAnd (T.eq (df ?? "MPCM21_region") 2)
                            (T.eq (df ?? "MPCS11_region") 2)
            -- T.logicalAnd (T.lt (df ?? "gm") 0.0) (T.gt (df ?? "pm") 0.0)
            -- T.lt (df ?? "gm") 25.0

-- | ((Mcm13+Mcm12+Mcm13)*Wcm1*Lcm1)+((Mdp11+Mdp12)*Wdp1*Ldp1)+((Mcm21+Mcm22)*Wcm2*Lcm2)+(Mcs11*Wcs1*Lcs1)+(Mres1*Wres*Lres)+(Lcap*Wcap*Mcap1)
area :: DF.DataFrame T.Tensor -> T.Tensor
area df = ((mcm11 + mcm12 + mcm13) * wcm1 * lcm1)
        + ((mdp11 + mdp12) * wdp1 * ldp1)
        + ((mcm21 + mcm22) * wcm2 * lcm2)
        + (mcs11 * wcs1 * lcs1)
        + (mres1 * wres * lres)
        + (mcap1 * wcap * lcap)
  where
    lcm1  = df ?? "Lcm1"
    lcm2  = df ?? "Lcm2"
    ldp1  = df ?? "Ldp1"
    lcs1  = df ?? "Lcs1"
    lcap  = df ?? "Lcap"
    lres  = df ?? "Lres"
    wcm1  = df ?? "Wcm1"
    wcm2  = df ?? "Wcm2"
    wdp1  = df ?? "Wdp1"
    wcs1  = df ?? "Wcs1"
    wcap  = df ?? "Wcap"
    wres  = df ?? "Wres"
    mcm11 = df ?? "Mcm11"
    mcm12 = df ?? "Mcm12"
    mcm13 = df ?? "Mcm13"
    mcm21 = df ?? "Mcm21"
    mcm22 = df ?? "Mcm22"
    mdp11 = df ?? "Mdp11"
    mdp12 = df ?? "Mdp12"
    mcs11 = df ?? "Mcs11"
    mcap1 = df ?? "Mcap1"
    mres1 = df ?? "Mres1"
