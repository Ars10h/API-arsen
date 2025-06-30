// // import { useState, useEffect } from "react";

// // export default function App() {
// //   const [status, setStatus] = useState(null);
// //   const [loading, setLoading] = useState(false);
// //   const [defense, setDefense] = useState("none");

// //   const fetchStatus = async () => {
// //     const res = await fetch("http://localhost:8000/status");
// //     const data = await res.json();
// //     setStatus(data);
// //   };

// //   const startTraining = async () => {
// //     setLoading(true);
// //     await fetch(`http://localhost:8000/train?defense=${defense}`, { method: "POST" });
// //     setLoading(false);
// //   };

// //   useEffect(() => {
// //     const interval = setInterval(fetchStatus, 1000);
// //     return () => clearInterval(interval);
// //   }, []);

// //   return (
// //     <div className="p-6 max-w-3xl mx-auto font-sans">
// //       <h1 className="text-3xl font-bold mb-4 text-blue-700">Federated Learning - MIA Tracker</h1>

// //       <div className="mb-4">
// //         <label className="mr-2 font-semibold">Select Defense:</label>
// //         <select
// //           className="border rounded px-2 py-1"
// //           value={defense}
// //           onChange={(e) => setDefense(e.target.value)}
// //         >
// //           <option value="none">None</option>
// //           <option value="dropout">Dropout</option>
// //           <option value="dp">Differential Privacy</option>
// //         </select>
// //       </div>

// //       <button
// //         className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded mb-6 disabled:bg-gray-400"
// //         onClick={startTraining}
// //         disabled={loading || (status && status.status === "running")}
// //       >
// //         {loading ? "Starting..." : "Start Training"}
// //       </button>

// //       {status && (
// //         <div className="bg-gray-100 p-4 rounded shadow">
// //           <p><strong>Status:</strong> {status.status}</p>
// //           <p><strong>Round:</strong> {status.current_round} / {status.total_rounds}</p>
// //           <p><strong>Message:</strong> {status.message}</p>

// //           <div className="mt-4">
// //             <h2 className="font-semibold mb-2">Metrics:</h2>
// //             <div className="space-y-1">
// //               {status.accuracy.map((acc, idx) => (
// //                 <div key={idx} className="flex justify-between">
// //                   <span>Round {idx + 1}</span>
// //                   <span>Accuracy: {acc.toFixed(2)}%</span>
// //                   <span>MIA AUC: {status.mia_auc[idx].toFixed(2)}%</span>
// //                 </div>
// //               ))}
// //             </div>
// //           </div>
// //         </div>
// //       )}
// //     </div>
// //   );
// // }

// import { useEffect, useState } from "react";

// export default function App() {
//   const [status, setStatus] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [defense, setDefense] = useState("none");

//   const fetchStatus = async () => {
//     try {
//       const res = await fetch("http://localhost:8000/status");
//       const data = await res.json();
//       setStatus(data);
//     } catch (err) {
//       console.error("Erreur lors de la récupération du status :", err);
//     }
//   };

//   const startTraining = async () => {
//     setLoading(true);
//     try {
//       await fetch(`http://localhost:8000/train?defense=${defense}`, {
//         method: "POST",
//       });
//     } catch (err) {
//       console.error("Erreur lors du démarrage de l'entraînement :", err);
//     } finally {
//       setLoading(false);
//     }
//   };

//   useEffect(() => {
//     const interval = setInterval(fetchStatus, 1000);
//     return () => clearInterval(interval);
//   }, []);

//   return (
//     <div className="min-h-screen bg-gray-100 p-8">
//       <div className="max-w-3xl mx-auto bg-white shadow rounded p-6">
//         <h1 className="text-3xl font-bold mb-4 text-center text-blue-700">
//           Federated Learning - MIA Tracker
//         </h1>

//         <div className="flex items-center gap-4 mb-4">
//           <label className="font-medium text-gray-700">Defense:</label>
//           <select
//             className="border rounded px-3 py-1"
//             value={defense}
//             onChange={(e) => setDefense(e.target.value)}
//             disabled={loading || status?.status === "running"}
//           >
//             <option value="none">None</option>
//             <option value="dropout">Dropout</option>
//             <option value="dp">Differential Privacy</option>
//           </select>
//           <button
//             className={`px-4 py-2 rounded text-white ${
//               loading || status?.status === "running"
//                 ? "bg-gray-400 cursor-not-allowed"
//                 : "bg-blue-600 hover:bg-blue-700"
//             }`}
//             onClick={startTraining}
//             disabled={loading || status?.status === "running"}
//           >
//             {loading ? "Starting..." : "Start Training"}
//           </button>
//         </div>

//         {status ? (
//           <div>
//             <p><strong>Status:</strong> {status.status}</p>
//             <p><strong>Round:</strong> {status.current_round} / {status.total_rounds}</p>
//             <p><strong>Message:</strong> {status.message}</p>

//             {status.accuracy?.length > 0 && status.mia_auc?.length > 0 && (
//               <div className="mt-6">
//                 <h2 className="font-semibold text-lg mb-2">Metrics per Round</h2>
//                 <div className="space-y-2">
//                   {status.accuracy.map((acc, idx) => {
//                     const mia = status.mia_auc[idx];
//                     if (mia === undefined) return null;

//                     return (
//                       <div
//                         key={idx}
//                         className="bg-gray-100 px-4 py-2 rounded flex justify-between items-center text-sm"
//                       >
//                         <span>Round {idx + 1}</span>
//                         <span>Accuracy: {acc.toFixed(2)}%</span>
//                         <span>MIA AUC: {mia.toFixed(2)}%</span>
//                       </div>
//                     );
//                   })}
//                 </div>
//               </div>
//             )}

//             {(status.status === "done" || status.accuracy.length > 0) && (
//               <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
//                 <div>
//                   <h3 className="font-medium mb-1">Accuracy / AUC Curve</h3>
//                   <img
//                     src="http://localhost:8000/plot/miacurve"
//                     alt="MIA Curve"
//                     className="w-full rounded border"
//                   />
//                 </div>
//                 <div>
//                   <h3 className="font-medium mb-1">ROC Curve</h3>
//                   <img
//                     src="http://localhost:8000/plot/miaroc"
//                     alt="MIA ROC"
//                     className="w-full rounded border"
//                   />
//                 </div>
//               </div>
//             )}
//           </div>
//         ) : (
//           <p className="text-sm text-gray-500">En attente de l'état de l'entraînement...</p>
//         )}
//       </div>
//     </div>
//   );
// }

import { useEffect, useState } from "react";

export default function App() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [defense, setDefense] = useState("none");

  const fetchStatus = async () => {
    try {
      const res = await fetch("http://localhost:8000/status");
      const data = await res.json();
      setStatus(data);
    } catch (err) {
      console.error("Erreur lors de la récupération du status :", err);
    }
  };

  const startTraining = async () => {
    setLoading(true);
    try {
      await fetch(`http://localhost:8000/train?defense=${defense}`, {
        method: "POST",
      });
    } catch (err) {
      console.error("Erreur lors du démarrage de l'entraînement :", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, []);

  const isTrainingDone = status?.status === "done";

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white p-8">
      <div className="max-w-4xl mx-auto bg-white shadow-md rounded-lg p-8">
        <h1 className="text-4xl font-bold text-center text-blue-800 mb-6">
          Federated Learning - MIA Tracker
        </h1>

        <div className="flex flex-wrap items-center justify-center gap-4 mb-6">
          <label className="font-medium text-gray-700">Defense:</label>
          <select
            className="border rounded px-3 py-2"
            value={defense}
            onChange={(e) => setDefense(e.target.value)}
            disabled={loading || status?.status === "running"}
          >
            <option value="none">None</option>
            <option value="dropout">Dropout</option>
            <option value="dp">Differential Privacy</option>
          </select>
          <button
            className={`px-4 py-2 rounded text-white transition ${
              loading || status?.status === "running"
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700"
            }`}
            onClick={startTraining}
            disabled={loading || status?.status === "running"}
          >
            {loading ? "Starting..." : "Start Training"}
          </button>
        </div>

        {status ? (
          <>
            <div className="mb-6 text-sm text-gray-700">
              <p>
                <strong>Status:</strong> {status.status}
              </p>
              <p>
                <strong>Round:</strong> {status.current_round} / {status.total_rounds}
              </p>
              <p>
                <strong>Message:</strong> {status.message}
              </p>
            </div>

            {status.accuracy.length > 0 && (
              <div className="mb-8">
                <h2 className="text-xl font-semibold mb-4 text-blue-700">Metrics per Round</h2>
                <div className="space-y-2">
                  {status.accuracy.map((acc, idx) => {
                    const mia = status.mia_auc[idx];
                    if (mia === undefined) return null;

                    return (
                      <div
                        key={idx}
                        className="bg-blue-50 border border-blue-200 px-4 py-2 rounded flex justify-between items-center text-sm"
                      >
                        <span className="font-medium">Round {idx + 1}</span>
                        <span>Accuracy: {acc.toFixed(2)}%</span>
                        <span>MIA AUC: {mia.toFixed(2)}%</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {isTrainingDone && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Accuracy / AUC Curve</h3>
                  <img
                    src="http://localhost:8000/plot/miacurve"
                    alt="MIA Curve"
                    className="w-full h-auto rounded border"
                  />
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-2">ROC Curve</h3>
                  <img
                    src="http://localhost:8000/plot/miaroc"
                    alt="MIA ROC"
                    className="w-full h-auto rounded border"
                  />
                </div>
              </div>
            )}
          </>
        ) : (
          <p className="text-gray-500 text-center">Chargement de l'état du serveur...</p>
        )}
      </div>
    </div>
  );
}
