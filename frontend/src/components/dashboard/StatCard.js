import React from "react";
import styles from "./StatCard.module.css";

const StatCard = ({
  title,
  value,
  delta,
  deltaPositive,
  bgColor,
  textColor,
}) => {
  return (
    <div
      className={styles.card}
      style={{
        backgroundColor: bgColor || "white",
        color: textColor || "#111827",
      }}
    >
      <h4 className={styles.title}>{title}</h4>
      <div className={styles.valueWrap}>
        <span className={styles.value}>{value}</span>
        <span
          className={`${styles.deltaSmall} ${
            deltaPositive ? styles.positive : styles.negative
          }`}
        >
          {delta} {deltaPositive ? "↗" : "↘"}
        </span>
      </div>
    </div>
  );
};

export default StatCard;
