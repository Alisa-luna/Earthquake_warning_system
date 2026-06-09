#ifndef EARTHQUAKE_STATE_MACHINE_H
#define EARTHQUAKE_STATE_MACHINE_H

#include <Arduino.h>
#include "config.h"

enum class AlertLevel { 
    NONE = 0, 
    YELLOW = 1, 
    ORANGE = 2, 
    RED = 3 
};

class EarthquakeStateMachine {
private:
    AlertLevel currentLevel;
    uint32_t eventStartTime;
    float lastProbability;
    int consecutiveTriggers;
    
public:
    EarthquakeStateMachine() { 
        reset(); 
    }
    
    void reset() {
        currentLevel = AlertLevel::NONE;
        eventStartTime = 0;
        lastProbability = 0;
        consecutiveTriggers = 0;
    }
    
    // 第一级检测：Z轴置信度超过阈值
    AlertLevel firstLevelCheck(float probZ) {
        if (probZ > YELLOW_THRESHOLD) {
            consecutiveTriggers++;
        } else {
            consecutiveTriggers = 0;
        }
        
        if (currentLevel == AlertLevel::NONE && consecutiveTriggers >= 3) {
            currentLevel = AlertLevel::YELLOW;
            eventStartTime = millis();
            lastProbability = probZ;
            Serial.println("🟡 [StateMachine] Level → YELLOW");
            return AlertLevel::YELLOW;
        }
        
        return currentLevel;
    }
    
    // 第二级检测：三轴交叉验证
    AlertLevel secondLevelCheck(float probX, float probY, float probZ) {
        if (currentLevel != AlertLevel::YELLOW) {
            return currentLevel;
        }
        
        if (millis() - eventStartTime > 30000) {
            Serial.println("⏰ [StateMachine] YELLOW timeout, resetting");
            reset();
            return AlertLevel::NONE;
        }
        
        int votes = 0;
        if (probX > ORANGE_THRESHOLD) votes++;
        if (probY > ORANGE_THRESHOLD) votes++;
        if (probZ > ORANGE_THRESHOLD) votes++;
        
        lastProbability = (probX + probY + probZ) / 3.0f;
        
        if (votes >= VOTE_COUNT) {
            currentLevel = AlertLevel::ORANGE;
            Serial.printf("🟠 [StateMachine] Level → ORANGE (votes: %d/3, prob: %.3f)\n", 
                          votes, lastProbability);
            return AlertLevel::ORANGE;
        }
        
        return currentLevel;
    }
    
    // 最终确认
    AlertLevel finalConfirm() {
        if (currentLevel == AlertLevel::ORANGE) {
            currentLevel = AlertLevel::RED;
            Serial.println("🔴 [StateMachine] Level → RED (CONFIRMED)");
            return AlertLevel::RED;
        }
        return currentLevel;
    }
    
    AlertLevel getLevel() { return currentLevel; }
    float getLastProbability() { return lastProbability; }
    uint32_t getEventDuration() { 
        return (eventStartTime > 0) ? (millis() - eventStartTime) : 0; 
    }
    
    float calculateMagnitude(float maxAccelG) {
        if (maxAccelG < 0.001f) return 0;
        float pga_cm_s2 = maxAccelG * 980.0f;
        return log10(pga_cm_s2) * 0.8f + 1.5f;
    }
    
    int calculateIntensity(float maxAccelG) {
        float cm_s2 = maxAccelG * 980.0f;
        if (cm_s2 < 0.31) return 1;
        if (cm_s2 < 0.63) return 2;
        if (cm_s2 < 1.25) return 3;
        if (cm_s2 < 2.50) return 4;
        if (cm_s2 < 5.00) return 5;
        if (cm_s2 < 10.0) return 6;
        if (cm_s2 < 25.0) return 7;
        if (cm_s2 < 50.0) return 8;
        if (cm_s2 < 100.0) return 9;
        if (cm_s2 < 250.0) return 10;
        return 11;
    }
};

#endif