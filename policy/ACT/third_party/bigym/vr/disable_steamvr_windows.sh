#!/bin/bash
# Fix for non-closing SteamVR windows: https://github.com/ValveSoftware/SteamVR-for-Linux/issues/577
# Run it after installing SteamVR
sed -ri 's/"preload"(.*)true/"preload"\1false/g' $HOME/.steam/steam/steamapps/common/SteamVR/drivers/lighthouse/resources/webhelperoverlays.json
sed -ri 's/"preload"(.*)true/"preload"\1false/g' $HOME/.steam/steam/steamapps/common/SteamVR/resources/webhelperoverlays.json
sed -ri 's/"_preload"(.*)true/"_preload"\1false/g' $HOME/.steam/steam/steamapps/common/SteamVR/drivers/vrlink/resources/webhelperoverlays.json
