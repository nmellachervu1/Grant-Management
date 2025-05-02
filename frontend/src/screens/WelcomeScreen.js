"use client";

import React from "react";
import styles from "./WelcomeScreen.module.css";

const WelcomeScreen = ({ onStart }) => {
  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Welcome to Obligation Insights</h1>
      <p className={styles.subtitle}>Tap below to get started</p>
      <div className={styles.buttonRow}>
        <button
          onClick={() => onStart("global")}
          className={styles.button}
        >
          Global Portfolio
        </button>
        <button
          onClick={() => onStart("country")}
          className={styles.button}
        >
          Country Portfolio
        </button>
        <button
          onClick={() => onStart("grant")}
          className={styles.button}
        >
          Grant Tool
        </button>
      </div>
    </div>
  );
};

export default WelcomeScreen;
