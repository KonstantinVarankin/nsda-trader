import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import config from '../config';

ChartJS.register(ArcElement, Tooltip, Legend);

const MarketSentiment = () => {
  const [sentimentData, setSentimentData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [debugInfo, setDebugInfo] = useState('');

  useEffect(() => {
    const fetchSentimentData = async () => {
      setLoading(true);
      setDebugInfo('Starting to fetch data...');
      try {
        const url = `${config.API_URL}${config.API_V1_STR}/market-sentiment`;
        setDebugInfo(prevInfo => prevInfo + `\nFetching from URL: ${url}`);
        console.log('Fetching sentiment data from:', url);

        const response = await axios.get(url, { timeout: 15000 });
        setDebugInfo(prevInfo => prevInfo + `\nReceived response with status: ${response.status}`);
        console.log('Received response:', response.data);

        if (response.data) {
          setSentimentData(response.data);
          setDebugInfo(prevInfo => prevInfo + '\nData successfully set');
        } else {
          setDebugInfo(prevInfo => prevInfo + '\nResponse data is empty');
        }
        setError(null);
      } catch (error) {
        console.error('Error fetching market sentiment data:', error);
        setDebugInfo(prevInfo => prevInfo + `\nError occurred: ${error.message}`);
        if (error.response) {
          setDebugInfo(prevInfo => prevInfo + `\nResponse status: ${error.response.status}`);
          setDebugInfo(prevInfo => prevInfo + `\nResponse data: ${JSON.stringify(error.response.data)}`);
        }
        setError(error.response?.data?.detail || error.message || 'An error occurred while fetching data');
        setSentimentData(null);
      } finally {
        setLoading(false);
        setDebugInfo(prevInfo => prevInfo + '\nFetch attempt completed');
      }
    };

    fetchSentimentData();
    const interval = setInterval(fetchSentimentData, 60000); // Update every minute
    return () => clearInterval(interval);
  }, []);

  const renderPieChart = () => {
    if (!sentimentData) return null;

    const data = {
      labels: ['Positive', 'Negative', 'Neutral'],
      datasets: [
        {
          data: [sentimentData.positive, sentimentData.negative, sentimentData.neutral],
          backgroundColor: ['#36A2EB', '#FF6384', '#FFCE56'],
          hoverBackgroundColor: ['#36A2EB', '#FF6384', '#FFCE56'],
        },
      ],
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
    };

    return <Pie data={data} options={options} />;
  };

  if (loading) {
    return (
      <div>
        <p>Loading market sentiment data...</p>
        <pre>{debugInfo}</pre>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <p>Error: {error}</p>
        <pre>{debugInfo}</pre>
      </div>
    );
  }

  if (!sentimentData) {
    return (
      <div>
        <p>No sentiment data available.</p>
        <pre>{debugInfo}</pre>
      </div>
    );
  }

  return (
    <div className="market-sentiment">
      <h2>Market Sentiment Analysis</h2>
      <div style={{ width: '300px', height: '300px' }}>
        {renderPieChart()}
      </div>
      <div>
        <p>Positive: {sentimentData.positive}%</p>
        <p>Negative: {sentimentData.negative}%</p>
        <p>Neutral: {sentimentData.neutral}%</p>
      </div>
      <pre>{debugInfo}</pre>
    </div>
  );
};

export default MarketSentiment;