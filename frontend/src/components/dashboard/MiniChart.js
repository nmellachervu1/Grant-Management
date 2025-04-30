"use client";

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import styles from "./MiniChart.module.css";

// Sample data
const data = [
  { name: "Jan", context: 10, content: 20, contact: 15 },
  { name: "Feb", context: 12, content: 22, contact: 13 },
  { name: "Mar", context: 14, content: 25, contact: 18 },
  { name: "Apr", context: 16, content: 20, contact: 20 },
  { name: "May", context: 18, content: 30, contact: 24 },
  { name: "Jun", context: 20, content: 28, contact: 27 },
];

const MiniChart = () => {
  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <span className={styles.title}>CHART TITLE</span>
        <span className={styles.subtext}>This Week ⏷</span>
      </div>

      <div className={styles.chartArea}>
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={data}>
            <XAxis dataKey="name" hide />
            <Tooltip />
            <Line type="monotone" dataKey="context" stroke="#4ade80" strokeWidth={2} />
            <Line type="monotone" dataKey="content" stroke="#60a5fa" strokeWidth={2} />
            <Line type="monotone" dataKey="contact" stroke="#facc15" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className={styles.legend}>
        <span style={{ color: "#4ade80" }}>●</span> Context
        <span style={{ color: "#60a5fa" }}>●</span> Content
        <span style={{ color: "#facc15" }}>●</span> Contact
      </div>
    </div>
  );
};

export default MiniChart;
