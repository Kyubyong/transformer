#!/bin/bash

mkdir iwslt2016 | wget -qO- --show-progress https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz | tar xz; mv de-en iwslt2016