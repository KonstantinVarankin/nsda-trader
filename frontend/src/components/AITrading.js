import React, { useState } from 'react';
import axios from 'axios';

function AITrading() {
    const [decision, setDecision] = useState(null);

    const startAITrading = async () => {
        try {
            const response = await axios.post('/api/start-ai-trading');
            setDecision(response.data.trading_decision);
        } catch (error) {
            console.error('Error starting AI trading:', error);
        }
    };

    return (
        <div>
            <h2>AI Trading</h2>
            <button onClick={startAITrading}>Start AI Trading</button>
            {decision && <p>Trading Decision: {decision}</p>}
        </div>
    );
}

export default AITrading;