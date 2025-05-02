"use client";

import React from "react";
import styles from "./SummaryCard.module.css";

const SummaryCard = ({ summary, isLoading }) => {
  return (
    <div className={styles.card}>
      <h4 className={styles.title}>Summary</h4>
      <p className={styles.text}>
        {isLoading
          ? "LLM Response is Being Generated..."
          : summary || "No summary available."}
      </p>
    </div>
  );
};

export default SummaryCard;

