"use client";

import React, { useState } from "react";
import WelcomeScreen from "@/screens/WelcomeScreen";
import ObligationInsights from "@/screens/ObligationInsights";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient();

export default function Home() {
  const [selectedScreen, setSelectedScreen] = useState(null);

  return (
    <QueryClientProvider client={queryClient}>
      {!selectedScreen && <WelcomeScreen onStart={setSelectedScreen} />}

      {selectedScreen === "global" && <ObligationInsights mode="global" />}
      {selectedScreen === "country" && <ObligationInsights mode="country" />}
      {selectedScreen === "grant" && <div>Grant Tool coming soon</div>}
    </QueryClientProvider>
  );
}


