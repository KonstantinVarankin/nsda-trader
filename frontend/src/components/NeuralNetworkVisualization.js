import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const NeuralNetworkVisualization = ({ layers }) => {
  const svgRef = useRef();

  useEffect(() => {
    if (!layers || layers.length === 0) return;

    const width = 800;
    const height = 600;
    const nodeRadius = 20;

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    svg.selectAll('*').remove();

    const xScale = d3.scaleLinear()
      .domain([0, layers.length - 1])
      .range([100, width - 100]);

    layers.forEach((layer, layerIndex) => {
      const yScale = d3.scaleLinear()
        .domain([0, layer - 1])
        .range([50, height - 50]);

      // Draw nodes
      svg.selectAll(`.node-layer-${layerIndex}`)
        .data(d3.range(layer))
        .enter()
        .append('circle')
        .attr('class', `node-layer-${layerIndex}`)
        .attr('cx', xScale(layerIndex))
        .attr('cy', d => yScale(d))
        .attr('r', nodeRadius)
        .attr('fill', 'lightblue')
        .attr('stroke', 'blue');

      // Draw connections to next layer
      if (layerIndex < layers.length - 1) {
        const nextLayer = layers[layerIndex + 1];
        const nextYScale = d3.scaleLinear()
          .domain([0, nextLayer - 1])
          .range([50, height - 50]);

        for (let i = 0; i < layer; i++) {
          for (let j = 0; j < nextLayer; j++) {
            svg.append('line')
              .attr('x1', xScale(layerIndex))
              .attr('y1', yScale(i))
              .attr('x2', xScale(layerIndex + 1))
              .attr('y2', nextYScale(j))
              .attr('stroke', 'gray')
              .attr('stroke-width', 0.5);
          }
        }
      }
    });
  }, [layers]);

  return <svg ref={svgRef}></svg>;
};

export default NeuralNetworkVisualization;