"use client";

import React from "react";
import { Chart as ChartJS, LineElement, PointElement, Tooltip, LinearScale, Title, Filler, Legend, CategoryScale } from "chart.js";
import { Chart } from "react-chartjs-2";
import styles from "./PerformanceChart.module.css";

ChartJS.register(LineElement, PointElement, Tooltip, LinearScale, Title, Filler, Legend, CategoryScale);

const PerformanceChart = ({ data }) => {
  if (!data) return null;

  // Convert to X = liquidation year scale
  const scaleX = (val) => (val * 60) / 12 * 0.01;

  // Blue average line
  const avgLine = data.avg_line.GrantTimeElapsed.map((x, i) => ({
    x: scaleX(x),
    y: data.avg_line.ObligationSpent[i],
  }));

  // Red shaded area
  const redArea = data.area_data.GrantTimeElapsed.map((x, i) => ({
    x: scaleX(x),
    y: data.area_data.UDOPredictedLevel[i],
  }));

  // Scatter points
  const scatterPoints = data.latest_months_data.GrantTimeElapsed.map((x, i) => ({
    x: scaleX(x),
    y: data.latest_months_data.ObligationSpent[i],
    fullUniqueID: data.latest_months_data.UniqueID[i],
    grantee: data.latest_months_data.Grantee[i],
    country: data.latest_months_data.Country[i],
    monthsRemaining: data.latest_months_data.MonthsRemaining[i],
    backgroundColor: getRandomColor(), // individual point color
  }));

  function getRandomColor() {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let j = 0; j < 6; j++) {
      color += letters[Math.floor(Math.random() * 13)];
    }
    return color;
  }

  const chartData = {
    datasets: [
      {
        label: "Below 100% Liquidation Pattern",
        data: redArea,
        borderColor: "rgba(255, 99, 132, 0.6)",
        backgroundColor: "rgba(255, 99, 132, 0.3)",
        fill: true,
        pointRadius: 0,
        tension: 0.4,
        yAxisID: "y",
      },
      {
        label: "Historical Average Liquidation Rate",
        data: avgLine,
        borderColor: "#3b82f6",
        backgroundColor: "#3b82f6",
        fill: false,
        pointRadius: 0,
        tension: 0.4,
        yAxisID: "y",
      },
      {
        label: "Grants",
        data: scatterPoints,
        showLine: false,
        pointRadius: 6,
        backgroundColor: scatterPoints.map((p) => p.backgroundColor),
        parsing: {
          xAxisKey: "x",
          yAxisKey: "y",
        },
        yAxisID: "y",
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      tooltip: {
        callbacks: {
          label: function (context) {
            const d = context.raw;
            return [
              `UniqueID: ${d.fullUniqueID}`,
              `% Time Elapsed: ${d.x}`,
              `% Obligation Spent: ${d.y.toFixed(2)}`,
              `Months Remaining: ${d.monthsRemaining}`,
              `Grantee: ${d.grantee}`,
              `Country: ${d.country}`,
            ];
          },
        },
      },
      legend: {
        labels: {
          font: {
            size: 12,
          },
          filter: (legendItem) => legendItem.text !== "Grants",
        },
      },
    },
    scales: {
      x: {
        type: "linear",
        title: {
          display: true,
          text: "Liquidation Year",
          font: {
            size: 14,
          },
        },
        min: 0,
        max: 5,
        ticks: {
          callback: function (val) {
            return `Y${val}`;
          },
        },
      },
      y: {
        title: {
          display: true,
          text: "% of Obligation Liquidated",
          font: {
            size: 14,
          },
        },
        min: 0,
        max: 100,
      },
    },
  };

  return (
    <div className={styles.chartWrapper}>
      <h3 className={styles.chartTitle}>Current Progress vs. Liquidation Patterns</h3>
      <div className={styles.chartContainer}>
        <Chart type="line" data={chartData} options={chartOptions} />
      </div>
    </div>
  );
};

export default PerformanceChart;

