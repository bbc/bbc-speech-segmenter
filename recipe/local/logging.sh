#!/bin/bash

# Copyright  2021 Matt Haynes <matt.haynes@bbc.co.uk>
#            2021 Misa Ogura <misa.ogura@bbc.co.uk>
# Apache 2.0

function log {
  local level=$1
  local stage=$2
  local message=$3
  local timestamp="$(date "+%Y-%m-%dT%H:%M:%S.%3NZ" --utc)"

  echo "$timestamp $level $stage $message" >&2
}

function debug {
  log "DEBUG" "$1" "$2"
}

function info {
  log "INFO" "$1" "$2"
}

function warn {
  log "WARN" "$1" "$2"
}

function error {
  log "ERROR" "$1" "$2"
}

function fatal {
  log "FATAL" "$1" "$2"
}

function fail {
  fatal "$1" "$2"
  exit 1
}
