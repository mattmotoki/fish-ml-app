import React, { Component } from 'react';

import { Model } from './Model'
import { modelConfigs } from './modelConfigs';

export class App extends Component {

  constructor() {
    super()
    this.imageRef = React.createRef();
    this.state = {
      image: null, annotationGroups: [[]]
    }
    this.handleFile = this.handleFile.bind(this)
    this.handlePredictions = this.handlePredictions.bind(this)
  }

  handlePredictions(index, anns) {
    let annotationGroups = this.state.annotationGroups
    annotationGroups[index] = anns
    this.setState({ annotationGroups })
  }

  handleFile(evt) {
    const image = URL.createObjectURL(evt.target.files[0]);
    const annotationGroups = [[]]
    this.setState({ image, annotationGroups })
  }
  render() {
    return (
      <div style={{ width: "500px", margin: "0 auto" }}>
        <h1>RetinaNetJS Object Detection</h1>
        <p>
          This page demonstrates use of the
          <a href="https://www.npmjs.com/package/retinanetjs">retinanetjs</a> package
          for deploying <a href="https://github.com/fizyr/keras-retinanet">keras-retinanet</a> models for inference in the browser. To use,
          simply:
        </p>
        <ol>
          <li>Enable one or more models (note the download sizes can be significant).</li>
          <li>Hit predict. Note that the inference all occurs in the browser using <a href="https://js.tensorflow.org">tensorflowjs</a>, so no data is sent to a server. Each detector's predictions appear in a different color.</li>
        </ol>
        <div>
          {
            modelConfigs.map((modelConfig, index) => {
              return (
                <Model
                  predictionCallback={(anns) => { this.handlePredictions(index, anns) }}
                  modelConfig={modelConfig}
                  imageRef={this.imageRef}
                  key={modelConfig.modelUrl}
                ></Model>
              )
            })
          }
        </div>
        <div>
          <input type="file" onChange={this.handleFile} id="file" />
        </div>
        <div style={{ position: "relative" }}>
          <img style={{ "width": "100%" }} ref={this.imageRef} src={this.state.image || "dog.jpg"} alt="your labeled upload" />
          {
            this.state.annotationGroups.map((annotationGroup, groupIndex) => {
              return (
                <div key={groupIndex}>
                  {
                    annotationGroup.map((annotation, annotationIndex) => {
                      return <div
                        key={annotationIndex}
                        style={{
                          position: "absolute",
                          border: `5px solid ${modelConfigs[groupIndex].color}`,
                          color: modelConfigs[groupIndex].color,
                          width: `${(annotation.x2 - annotation.x1) * 100}%`,
                          height: `${(annotation.y2 - annotation.y1) * 100}%`,
                          left: `${annotation.x1 * 100}%`,
                          top: `${annotation.y1 * 100}%`,
                          fontWeight: "bold"
                        }}><span style={{ padding: "5px" }}>{annotation.label}</span></div>
                    })
                  }
                </div>
              )
            })
          }
        </div>
      </div >

    )
  }
}



export default App;
