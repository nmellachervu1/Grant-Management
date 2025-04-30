"use client";

import React, { useState } from "react";
import FilterDropdown from "../components/dashboard/FilterDropdown";
import PerformanceChart from "../components/dashboard/PerformanceChart";
import SummaryCard from "../components/dashboard/SummaryCard";
import StatCard from "../components/dashboard/StatCard";
import MiniChart from "../components/dashboard/MiniChart";
import styles from "./ObligationInsights.module.css";

const ObligationInsights = () => {
  const [selectedArea, setSelectedArea] = useState("State-Specific");
  const [selectedTimeframe, setSelectedTimeframe] = useState("All");
  const [selectedVisualization, setSelectedVisualization] =
    useState("Graph Guide");

  return (
    <main className={styles.main}>
      {/* Filters */}
      <div className={styles.filtersRow}>
        <FilterDropdown
          label="USD Area Source"
          options={["State-Specific", "National"]}
          value={selectedArea}
          onChange={setSelectedArea}
        />
        <FilterDropdown
          label="Time Frame"
          options={["All", "Last 7 Days", "Last 30 Days", "Year to Date"]}
          value={selectedTimeframe}
          onChange={setSelectedTimeframe}
        />
        <FilterDropdown
          label="Visualization"
          options={["Graph Guide", "Bar Chart", "Line Chart"]}
          value={selectedVisualization}
          onChange={setSelectedVisualization}
        />
      </div>

      {/* Main content section — grey background */}
      <div className={styles.dashboardBackground}>
        <PerformanceChart />
        {/* All content below chart is inside a centered column */}
        <div className={styles.dashboardSection}>
          <SummaryCard />

          {/* middle stat cards go here later */}
          <div className={styles.statRow}>
            <StatCard
              title="Total Number of Grants"
              value="7,265"
              delta="+110%"
              deltaPositive={true}
              bgColor="#f9fafb"
              textColor="#111827"
            />
            <StatCard
              title="Expected Full Liquidation"
              value="3,671"
              delta="-0.25%"
              deltaPositive={false}
              bgColor="#f9fafb"
              textColor="#111827"
            />
            <StatCard
              title="Expected Full Liquidation"
              value="156"
              delta="-15.0%"
              deltaPositive={false}
              bgColor="#f9fafb"
              textColor="#111827"
            />
            <StatCard
              title="Expected Full Liquidation"
              value="2,318"
              delta="+0.08%"
              deltaPositive={true}
              bgColor="#f97316" /* orange */
              textColor="white"
            />
          </div>

          {/* bottom row stat cards + mini chart go here later */}
          <div className={styles.bottomRow}>
            <StatCard
              title="Total Obligation"
              value="156"
              delta="+13.1%"
              deltaPositive={true}
              bgColor="#f0abfc" // Tailwind pink-300
              textColor="#881337" // Deep pink/red
            />
            <StatCard
              title="Predicted Fallout"
              value="2,318"
              delta="+1.0%"
              deltaPositive={true}
              bgColor="#bfdbfe" // Tailwind blue-200
              textColor="#1e3a8a" // Dark blue
            />
            <MiniChart />
          </div>
        </div>
      </div>
    </main>
  );
};

export default ObligationInsights;
