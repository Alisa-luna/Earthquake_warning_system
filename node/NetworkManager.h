#ifndef NETWORK_MANAGER_H
#define NETWORK_MANAGER_H

#include <WiFi.h>

#define WIFI_RETRY_DELAY      10000
#define WIFI_RETRY_COUNT      5

class NetworkManager {
private:
    String wifi_ssid;
    String wifi_password;
    bool wifiConnected;
    int wifiRetryCount;
    unsigned long lastRetryTime;

public:
    NetworkManager() {
        wifi_ssid = "";
        wifi_password = "";
        wifiConnected = false;
        wifiRetryCount = 0;
        lastRetryTime = 0;
    }

    void setConfig(String ssid, String password) {
        wifi_ssid = ssid;
        wifi_password = password;
    }

    void setWiFiConfig(String ssid, String password) {
        wifi_ssid = ssid;
        wifi_password = password;
    }

    void begin() {
        if (wifi_ssid.length() == 0) {
            Serial.println("[WiFi] No config, skip");
            return;
        }
        WiFi.begin(wifi_ssid.c_str(), wifi_password.c_str());
        wifiRetryCount = 0;
        Serial.printf("[WiFi] Connecting to %s...\n", wifi_ssid.c_str());
    }

    bool update() {
        if (wifi_ssid.length() == 0) return false;

        if (WiFi.status() == WL_CONNECTED) {
            if (!wifiConnected) {
                wifiConnected = true;
                wifiRetryCount = 0;
                Serial.println("[WiFi] Connected!");
                Serial.printf("[WiFi] IP: %s\n", WiFi.localIP().toString().c_str());
            }
            return true;
        }

        if (wifiConnected) {
            wifiConnected = false;
            Serial.println("[WiFi] Disconnected!");
            lastRetryTime = millis();
        }

        if (millis() - lastRetryTime >= WIFI_RETRY_DELAY) {
            if (wifiRetryCount < WIFI_RETRY_COUNT) {
                wifiRetryCount++;
                Serial.printf("[WiFi] Reconnect %d/%d\n", wifiRetryCount, WIFI_RETRY_COUNT);
                WiFi.reconnect();
                lastRetryTime = millis();
            } else {
                wifiRetryCount = 0;
                Serial.println("[WiFi] Max retries, restarting...");
                WiFi.disconnect();
                delay(1000);
                WiFi.begin(wifi_ssid.c_str(), wifi_password.c_str());
                lastRetryTime = millis();
            }
        }

        return false;
    }

    bool isWiFi() { return WiFi.status() == WL_CONNECTED; }
    String getSSID() { return wifi_ssid; }
    String getPassword() { return wifi_password; }
};

#endif