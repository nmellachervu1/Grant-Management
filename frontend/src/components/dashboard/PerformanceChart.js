"use client";

import React from "react";
import {
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import styles from "./PerformanceChart.module.css";

// Dummy data for now
const data = [
  { name: "5k", value: 20 },
  { name: "10k", value: 35 },
  { name: "15k", value: 40 },
  { name: "20k", value: 25 },
  { name: "25k", value: 60 },
  { name: "30k", value: 45 },
  { name: "35k", value: 70 },
  { name: "40k", value: 50 },
  { name: "45k", value: 65 },
];

const PerformanceChart = () => {
  return (
    <div className={styles.chartWrapper}>
      <h3 className={styles.chartTitle}>
        Florida Portfolio Latest Month Spending Performance
      </h3>

      <div className={styles.legend}>
        <div className={styles.legendItem}>
          <div className={`${styles.dot} ${styles.greenDot}`}></div>
          <span>Spending Performance</span>
        </div>
        <div className={styles.legendItem}>
          <div className={`${styles.dot} ${styles.blueDot}`}></div>
          <span>Guide</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorArea" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#4ade80" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#4ade80" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Area
            type="monotone"
            dataKey="value"
            stroke="#4ade80"
            fillOpacity={1}
            fill="url(#colorArea)"
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#3b82f6"
            dot={{ r: 4 }}
            activeDot={{ r: 6 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PerformanceChart;
