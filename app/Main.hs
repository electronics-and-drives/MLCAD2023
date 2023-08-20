{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import MLCAD23
import Options.Applicative

main :: IO ()
main = execParser opts >>= run
  where
    desc = "MLCAD23 Codebase Snapthot"
    opts = info (args <**> helper) 
                (fullDesc <> progDesc desc <> header "MLCAD23")

-- | Command Line Argument Parser
args :: Parser Args
args = Args <$> strOption   ( long "mode" 
                           <> short 'm'
                           <> metavar "MODE" 
                           <> showDefault 
                           <> value "Sample"
                           <> help "Mode of operation" )
            <*> option auto ( long "parallel" 
                           <> short 'a'
                           <> metavar "INT" 
                           <> showDefault 
                           <> value 64
                           <> help "Number of parallel Spectre sessions" )
            <*> option auto ( long "points" 
                           <> short 'n'
                           <> metavar "INT" 
                           <> showDefault 
                           <> value 50000
                           <> help "Number of data points to sample" )
            <*> option auto ( long "runs" 
                           <> short 'r'
                           <> metavar "INT" 
                           <> showDefault 
                           <> value 10
                           <> help "Number of optimization runs" )
            <*> option auto ( long "epochs"
                           <> short 'e'
                           <> metavar "INT" 
                           <> showDefault 
                           <> value 300
                           <> help "Number of epochs to train for" )
            <*> strOption   ( long "optim" 
                           <> short 'o'
                           <> metavar "OPTIMIZER" 
                           <> showDefault 
                           <> value "Numeric"
                           <> help "Optimizer to use" )
            <*> strOption   ( long "ckt" 
                           <> short 'c'
                           <> metavar "CKT" 
                           <> showDefault 
                           <> value "sym"
                           <> help "Circuit ID" )
            <*> strOption   ( long "pdk" 
                           <> short 'p'
                           <> metavar "PDK" 
                           <> showDefault 
                           <> value "gpdk090"
                           <> help "PDK ID" )
            <*> strOption   ( long "corner"
                           <> short 'v'
                           <> metavar "CORNER"
                           <> showDefault
                           <> value "MC"
                           <> help "Process corner" )
