"use client";

import React, { useState } from 'react';
import StatCard from '../components/dashboard/StatCard';
import FilterDropdown from '../components/dashboard/FilterDropdown';

export default function Home() {
  const [selectedArea, setSelectedArea] = useState('State-Specific');
  const [selectedTimeframe, setSelectedTimeframe] = useState('All');
  const [selectedVisualization, setSelectedVisualization] = useState('Graph Guide');

  return (
    <main style={{ padding: '2rem' }}>
      {/* Filters Row */}
      <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '2rem', alignItems: 'flex-end' }}>
        <FilterDropdown
          label="USD Area Source"
          options={['State-Specific', 'National']}
          value={selectedArea}
          onChange={setSelectedArea}
        />
        <FilterDropdown
          label="Time Frame"
          options={['All', 'Last 7 Days', 'Last 30 Days', 'Year to Date']}
          value={selectedTimeframe}
          onChange={setSelectedTimeframe}
        />
        <FilterDropdown
          label="Visualization"
          options={['Graph Guide', 'Bar Chart', 'Line Chart']}
          value={selectedVisualization}
          onChange={setSelectedVisualization}
        />
      </div>

      {/* Stat Cards */}
      <div style={{ display: 'flex', gap: '1rem' }}>
        <StatCard
          title="Total Grants"
          value="345"
          delta="+5.3%"
          deltaPositive={true}
          bgColor="#f0f9ff"
          textColor="#0c4a6e"
        />
        <StatCard
          title="Predicted Fallout"
          value="120"
          delta="-2.3%"
          deltaPositive={false}
          bgColor="#fff7ed"
          textColor="#7c2d12"
        />
      </div>
    </main>
  );
}


