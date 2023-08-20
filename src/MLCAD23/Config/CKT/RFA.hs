{-# OPTIONS_GHC -Wall -fplugin Debug.Breakpoint #-}

{-# LANGUAGE TypeApplications #-}

module MLCAD23.Config.CKT.RFA where

import           Data.Frame                      ((??))
import qualified Data.Frame                as DF
import qualified Torch                     as T
import qualified Torch.Extensions          as T
import qualified Torch.Functional.Internal as T (negative)
import           MLCAD23.Util

-- import           Debug.Breakpoint

electricalParams :: [String]
electricalParams = [ "MNDP11_gmoverid", "MPDP21_gmoverid", "MNCM11_gmoverid", "MPCM21_gmoverid", "MNCM31_gmoverid"
                   , "MNLS11_gmoverid", "MPLS21_gmoverid" -- , "MNRF11_gmoverid", "MPRF21_gmoverid"
                   , "MNDP11_fug", "MPDP21_fug", "MNCM11_fug", "MPCM21_fug", "MNCM31_fug"
                   , "MNLS11_fug", "MPLS21_fug" -- , "MNRF11_fug", "MPRF21_fug"
                   , "MNCM13_id", "MNCM32_id" -- "MPCM25_id"
                   , "MNDP11_vds", "MPDP21_vds", "MNCM11_vds" , "MPCM21_vds", "MNCM31_vds"
                   , "MNLS11_vds", "MPLS21_vds" -- , "MNRF11_vds", "MPRF21_vds"
                   ]

geometricalParams :: [String]
geometricalParams = [ "Ldp1",  "Ldp2",  "Lcm1",  "Lcm2",  "Lcm3",  "Lls1",  "Lls2",  "Lrf1",  "Lrf2"
                    , "Wdp1",  "Wdp2",  "Wcm1",  "Wcm2",  "Wcm3",  "Wls1",  "Wls2",  "Wrf1",  "Wrf2"
                    , "Mdp11", "Mdp21", "Mcm11", "Mcm21", "Mcm31", "Mls11", "Mls22", "Mrf11", "Mrf21"
                    , "Mdp12", "Mdp22", "Mcm12", "Mcm22", "Mcm32", "Mls12", "Mls21"
                                      , "Mcm13", "Mcm23"
                                      , "Mcm14", "Mcm24"
                                               , "Mcm25" ]

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
    i3     = i0
    i4     = i0 --  * 0.5 -- T.abs $ elec ?? "MPCM23_id"
    i5     = i0 --  * 0.5 -- T.abs $ elec ?? "MNCM14_id"
    i1'    = T.abs $ elec ?? "MNCM13_id"
    i2'    = i1'
    i6'    = T.abs $ elec ?? "MNCM32_id"
    mmin   = T.ones' [num,1]
    mcm11  = T.ones' [num,1]
    mcm12  = mcm11 -- T.ones' [num,1]
    mcm13  = T.mul mcm11 . fst . T.maxDim (T.Dim 1) T.KeepDim . T.cat (T.Dim 1)
           . (:[mmin]) . T.round' 0 $ i1' / i0
    mcm14  = mcm11
    mcm21  = T.ones' [num,1]
    mcm22  = T.mul mcm21 . fst . T.maxDim (T.Dim 1) T.KeepDim . T.cat (T.Dim 1)
           . (:[mmin]) . T.round' 0 $ i2' / i3
    mcm23  = mcm21
    mcm24  = T.mul mcm21 . fst . T.maxDim (T.Dim 1) T.KeepDim . T.cat (T.Dim 1)
           . (:[mmin]) . T.round' 0 $ i6' / i3
    mcm25  = mcm24
    mcm31  = T.mulScalar @Float 2.0 $ T.ones' [num,1]
    mcm32  = mcm31
    mls11  = T.mulScalar @Float 1.0 $ T.ones' [num,1]
    mls12  = mls11
    mls21  = T.mulScalar @Float 1.0 $ T.ones' [num,1]
    mls22  = mls21
    mdp11  = T.mulScalar @Float 2.0 $ T.ones' [num,1]
    mdp12  = mdp11
    mdp21  = T.mulScalar @Float 2.0 $ T.ones' [num,1]
    mdp22  = mdp12
    mrf11  = T.ones' [num,1]
    mrf21  = T.ones' [num,1]
    i1     = T.subScalar @Float 0.5e-6 $ (i0 * mcm13) / mcm11
    i2     = (i3 * mcm22) / mcm21
    i12    = i1 / 2.0
    i22    = i2 / 2.0
    i6     = (i3 * mcm24) / mcm21
    i8     = i6
    i7     = (i6 - i22) + (i8 - i12)
    vw     = T.abs (elec ?? "MPLS21_vds") + T.abs (elec ?? "MNLS11_vds")
           + T.abs (elec ?? "MNCM31_vds") 
    vy     = T.abs (elec ?? "MNCM31_vds") 
    vcm1   = T.abs (elec ?? "MNDP11_vds") - vw
    vcm2   = T.abs (elec ?? "MNCM31_vds") + T.abs (elec ?? "MPDP21_vds")
    gmidR  = T.full' @Float [num,1] 5.0
    fugR   = T.full' @Float [num,1] (10.0 ** 9.1)
    vdsR   = T.full' @Float [num,1] 0.95
    cm1In  = T.cat (T.Dim 1) [ i0, elec ?? "MNCM11_gmoverid", elec ?? "MNCM11_fug"
                             , elec ?? "MNCM11_vds", vss ]
    cm2In  = T.cat (T.Dim 1) [ i3, elec ?? "MPCM21_gmoverid", elec ?? "MPCM21_fug"
                             , elec ?? "MPCM21_vds", vss ]
    cm3In  = T.cat (T.Dim 1) [ i8, elec ?? "MNCM31_gmoverid", elec ?? "MNCM31_fug"
                             , elec ?? "MNCM31_vds", vss ]
    ls1In  = T.cat (T.Dim 1) [ i7, elec ?? "MNLS11_gmoverid", elec ?? "MNLS11_fug"
                             , elec ?? "MNLS11_vds", - vy ]
    ls2In  = T.cat (T.Dim 1) [ i7, elec ?? "MPLS21_gmoverid", elec ?? "MPLS21_fug"
                             , elec ?? "MPLS21_vds", vdd - vw ]
    dp1In  = T.cat (T.Dim 1) [ i12, elec ?? "MNDP11_gmoverid", elec ?? "MNDP11_fug"
                             , elec ?? "MNDP11_vds", vcm1 ]
    dp2In  = T.cat (T.Dim 1) [ i22, elec ?? "MPDP21_gmoverid", elec ?? "MPDP21_fug"
                             , elec ?? "MPDP21_vds", vdd - vcm2 ]
    rf1In  = T.cat (T.Dim 1) [ i4, gmidR, fugR, vdsR, vss ]
    rf2In  = T.cat (T.Dim 1) [ i5, gmidR, fugR, T.negative vdsR, vss ]
    cm1Out = nmos cm1In
    cm2Out = pmos cm2In
    cm3Out = nmos cm3In
    dp1Out = nmos dp1In
    dp2Out = pmos dp2In
    rf1Out = nmos rf1In
    rf2Out = pmos rf2In
    ls1Out = nmos ls1In
    ls2Out = pmos ls2In
    cm1W   = T.select' 1 0 cm1Out / mcm11
    cm2W   = T.select' 1 0 cm2Out / mcm21
    cm3W   = T.select' 1 0 cm3Out / mcm31
    dp1W   = T.select' 1 0 dp1Out / mdp11
    dp2W   = T.select' 1 0 dp2Out / mdp21
    ls1W   = T.select' 1 0 ls1Out / mls11
    ls2W   = T.select' 1 0 ls2Out / mls21
    rf1W   = T.mulScalar @Float 0.25e-6 $ T.ones' [num,1]
    rf2W   = T.mulScalar @Float 0.25e-6 $ T.ones' [num,1]
    -- rf1W   = T.select' 1 0 rf1Out / mrf11
    -- rf2W   = T.select' 1 0 rf2Out / mrf21
    cm1L   = T.select' 1 1 cm1Out
    cm2L   = T.select' 1 1 cm2Out
    cm3L   = T.select' 1 1 cm3Out
    dp1L   = T.select' 1 1 dp1Out
    dp2L   = T.select' 1 1 dp2Out
    ls1L   = T.select' 1 1 ls1Out
    ls2L   = T.select' 1 1 ls2Out
    rf1L   = (T.select' 1 1 rf1Out * rf1W) / T.select' 1 0 rf1Out
    rf2L   = (T.select' 1 1 rf2Out * rf2W) / T.select' 1 0 rf2Out
    -- rf1L   = T.select' 1 1 rf1Out
    -- rf2L   = T.select' 1 1 rf2Out
    geom   = DF.DataFrame geometricalParams
           $ T.cat (T.Dim 1) [ dp1L,  dp2L,  cm1L,  cm2L,  cm3L,  ls1L,  ls2L,  rf1L,  rf2L
                             , dp1W,  dp2W,  cm1W,  cm2W,  cm3W,  ls1W,  ls2W,  rf1W,  rf2W
                             , mdp11, mdp21, mcm11, mcm21, mcm31, mls11, mls22, mrf11, mrf21
                             , mdp12, mdp22, mcm12, mcm22, mcm32, mls12, mls21
                                           , mcm13, mcm23
                                           , mcm14, mcm24
                                                  , mcm25 ]

-- "MNCM13_id", "MNCM32_id"
branchMultipliers :: [(Int, Int)]
branchMultipliers = [(1,4), (2,5)]

-- "MNDP11_vds", "MPDP21_vds", "MNCM11_vds" , "MPCM21_vds",
-- "MNCM31_vds" "MNLS11_vds", "MPLS21_vds"
nodeVoltageMultipliers :: [Float]
nodeVoltageMultipliers = [0.75, -0.75, 0.42, -0.45, 0.2, 0.38, -0.38]

trainMask :: DF.DataFrame T.Tensor -> T.Tensor
trainMask df = T.logicalAnd (T.gt (df ?? "pm")            0.0 )
             . T.logicalAnd (T.lt (df ?? "gm")            0.0 )
             $ T.logicalAnd (T.gt (df ?? "sr_r")      10000.0 )
                            (T.lt (df ?? "sr_f")    (-10000.0))
             --                 (T.lt (df ?? "voff_stat") 0.1)

-- | ((Mdp11+Mdp12)*Wdp1*Ldp1)+((Mdp21+Mdp22)*Wdp2*Ldp2)+((Mcm11+Mcm12+Mcm13+Mcm14)*Wcm1*Lcm1)+((Mcm21+Mcm22+Mcm23+Mcm24+Mcm25)*Wcm2*Lcm2)+((Mcm31+Mcm32)*Wcm3*Lcm3)+((Mls11+Mls21)*Wls1*Lls1)+(Mrf11*Wrf1*Lrf1)+((Mls21+Mls22)*Wls2*Lls2)+(Mrf21*Wrf2*Lrf2)
area :: DF.DataFrame T.Tensor -> T.Tensor
area df = ((mdp11 +mdp12) * wdp1 * ldp1)
        + ((mdp21 +mdp22) * wdp2 * ldp2)
        + ((mcm11 + mcm12 + mcm13 + mcm14) * wcm1 * lcm1)
        + ((mcm21 + mcm22 + mcm23 + mcm24 + mcm25) * wcm2 * lcm2)
        + ((mcm31 + mcm32) * wcm3 * lcm3)
        + ((mls11 + mls12) * wls1 * lls1)
        + ((mls21 + mls22) * wls2 * lls2)
        + (mrf11 * wrf1 * lrf1)
        + (mrf21 * wrf2 * lrf2)
  where
    ldp1  = df ?? "Ldp1"
    ldp2  = df ?? "Ldp2"
    lcm1  = df ?? "Lcm1"
    lcm2  = df ?? "Lcm2"
    lcm3  = df ?? "Lcm3"
    lls1  = df ?? "Lls1"
    lrf1  = df ?? "Lrf1"
    lls2  = df ?? "Lls2"
    lrf2  = df ?? "Lrf2"
    wdp1  = df ?? "Wdp1"
    wdp2  = df ?? "Wdp2"
    wcm1  = df ?? "Wcm1"
    wcm2  = df ?? "Wcm2"
    wcm3  = df ?? "Wcm3"
    wls1  = df ?? "Wls1"
    wrf1  = df ?? "Wrf1"
    wls2  = df ?? "Wls2"
    wrf2  = df ?? "Wrf2"
    mdp11 = df ?? "Mdp11"
    mdp12 = df ?? "Mdp12"
    mdp21 = df ?? "Mdp21"
    mdp22 = df ?? "Mdp22"
    mcm11 = df ?? "Mcm11"
    mcm12 = df ?? "Mcm12"
    mcm13 = df ?? "Mcm13"
    mcm14 = df ?? "Mcm14"
    mcm21 = df ?? "Mcm21"
    mcm22 = df ?? "Mcm22"
    mcm23 = df ?? "Mcm23"
    mcm24 = df ?? "Mcm24"
    mcm25 = df ?? "Mcm25"
    mcm31 = df ?? "Mcm31"
    mcm32 = df ?? "Mcm32"
    mls11 = df ?? "Mls11"
    mls12 = df ?? "Mls12"
    mrf11 = df ?? "Mrf11"
    mls21 = df ?? "Mls21"
    mls22 = df ?? "Mls22"
    mrf21 = df ?? "Mrf21"
