import React from 'react';
import StatCard from '../components/dashboard/StatCard';

export default function Home() {
  return (
    <main style={{ padding: '2rem', display: 'flex', gap: '1rem' }}>
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
    </main>
  );
}

