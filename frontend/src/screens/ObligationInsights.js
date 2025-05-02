"use client";

import React, { useState } from "react";
import FilterDropdown from "../components/dashboard/FilterDropdown";
import PerformanceChart from "../components/dashboard/PerformanceChart";
import SummaryCard from "../components/dashboard/SummaryCard";
import StatCard from "../components/dashboard/StatCard";
import MiniChart from "../components/dashboard/MiniChart";
import styles from "./ObligationInsights.module.css";
import { useQuery } from "@tanstack/react-query";
import axios from "axios";

const ObligationInsights = ({ mode }) => {
  const [selectedArea, setSelectedArea] = useState("State-Specific");
  const [selectedTimeframe, setSelectedTimeframe] = useState("All");
  const [selectedVisualization, setSelectedVisualization] = useState("Graph Guide");

  function generatePromptFromData(mode, data) {
    const isGlobal = mode === "global";
    const totalGrants = data.total_grants;
    const pointsBelow = data.num_below;
    const pointsAbove = totalGrants - pointsBelow;
  
    const grants = data.latest_months_data;
  
    const dataPoints = grants.UniqueID.map((id, i) => {
      const x = grants.GrantTimeElapsed[i];
      const y = grants.ObligationSpent[i];
      return `ID: ${id}, % Time Elapsed: ${x.toFixed(2)}, % Obligation Spent: ${y.toFixed(2)}`;
    }).join("\n");
  
    return `
  Data Overview:
  - Total Grants: ${totalGrants}
  - Points Above Pattern: ${pointsAbove}
  - Less Than %100 Percent Liquidation Pattern (Red Shaded Area): ${pointsBelow}
  
  Data Details:
  ${dataPoints}
  
  Additional Information:
  - The data being analyzed is ${isGlobal ? "global" : "country-specific"}.
  - Key observations should highlight how the points above and below the pattern reflect trends in ${isGlobal ? "global" : "country-specific"} obligations.
  - The Red Shaded Area lines are calculated using ${isGlobal ? "Global Area Data" : "Country Area Data"}.
    `;
  }
  

  const endpointMap = {
    global: "http://127.0.0.1:5000/api/global_portfolio",
    country: "http://127.0.0.1:5000/api/uganda_portfolio", 
    grant: "http://127.0.0.1:5000/api/grant_tool", 
  };
  
  const endpoint = endpointMap[mode];
  
  const { data, isLoading, isError } = useQuery({
    queryKey: ["portfolioData", mode],
    queryFn: () => axios.get(endpoint).then(res => res.data),
    enabled: !!endpoint, 
  });

  // for summary prompt 
  const { data: summaryData, isLoading: isSummaryLoading } = useQuery({
    queryKey: ["summary", mode],
    queryFn: async () => {
      const prompt = generatePromptFromData(mode, data); // We'll define this helper next
      const response = await axios.post("http://127.0.0.1:5000/generate-summary", { prompt });
      return response.data.summary;
    },
    enabled: !!data, // only run after the first data is fetched
  });

  if (isLoading) return <div>Loading...</div>;
  if (isError) return <div>Error loading data</div>;

  function calculatePointsAboveBelow(data) {
    const grants = data.latest_months_data;
    const redLine = data.area_data;
  
    let pointsBelow = 0;
  
    for (let i = 0; i < grants.UniqueID.length; i++) {
      const x = grants.GrantTimeElapsed[i] * 60 / 12 * 0.01;
      const y = grants.ObligationSpent[i];
  
      if (x >= 5 && y < 98) {
        pointsBelow++;
        continue;
      }
  
      let interpolatedY = null;
      for (let j = 0; j < redLine.GrantTimeElapsed.length - 1; j++) {
        const x1 = redLine.GrantTimeElapsed[j] * 60 / 12 * 0.01;
        const x2 = redLine.GrantTimeElapsed[j + 1] * 60 / 12 * 0.01;
  
        if (x1 <= x && x2 >= x) {
          const y1 = redLine.UDOPredictedLevel[j];
          const y2 = redLine.UDOPredictedLevel[j + 1];
          interpolatedY = y1 + ((y2 - y1) * (x - x1)) / (x2 - x1);
          break;
        }
      }
  
      if (interpolatedY !== null && y < interpolatedY) {
        pointsBelow++;
      }
    }
  
    const totalGrants = data.total_grants;
    const pointsAbove = totalGrants - pointsBelow;
  
    return { pointsAbove, pointsBelow };
  }
  
  const { pointsAbove, pointsBelow } = calculatePointsAboveBelow(data);
  const totalGrants = data.total_grants;


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
        <PerformanceChart data={data}  />
        {/* All content below chart is inside a centered column */}
        <div className={styles.dashboardSection}>
          <SummaryCard summary={summaryData} isLoading={isSummaryLoading} />

          {/* middle stat cards go here later */}
          <div className={styles.statRow}>
            <StatCard
              title="Total Number of Grants"
              value={data?.total_grants?.toLocaleString() ?? "—"}
              delta="+110%"
              deltaPositive={true}
              bgColor="#f9fafb"
              textColor="#111827"
            />
            <StatCard
              title="Expected Full Liquidation"
              value={pointsAbove.toLocaleString()}
              delta="-0.25%"
              deltaPositive={false}
              bgColor="#f9fafb"
              textColor="#111827"
            />
            <StatCard
              title="Expected Not Full Liquidation"
              value={pointsBelow.toLocaleString()}
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
