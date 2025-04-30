"use client";

import React from "react";
import styles from "./SummaryCard.module.css";

const SummaryCard = () => {
  return (
    <div className={styles.card}>
      <h4 className={styles.title}>Summary</h4>
      <p className={styles.text}>
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam eu turpis molestie, dictum est a,
        mattis tellus. Sed dignissim, metus nec fringilla accumsan, risus sem sollicitudin lacus,
        ut interdum tellus elit sed risus.
      </p>
    </div>
  );
};

export default SummaryCard;
