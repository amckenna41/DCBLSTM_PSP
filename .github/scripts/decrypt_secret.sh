#!/bin/sh

#create dir for secrets
mkdir $HOME/secrets
# Decrypt service account json file via gpg tool
# --batch to prevent interactive command
# --yes to assume "yes" for questions
gpg --quiet --batch --yes --decrypt --passphrase="$PASSPHRASE" \
--output $HOME/secrets/my_secret.json ./.github/secrets/account_key.json.gpg

#https://www.google.com/search?q=gpg+with+github+secrets&oq=gpg+with+github+secrets&aqs=chrome..69i57j0i22i30l2.6191j0j4&sourceid=chrome&ie=UTF-8
