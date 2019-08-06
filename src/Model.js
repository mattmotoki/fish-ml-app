import React, { Component } from 'react';

import { Line } from 'rc-progress'

import { load } from 'retinanetjs'

export class Model extends Component {
    constructor(props) {
        super(props);
        this.state = {
            detector: null,
            togglingModel: false,
            predicting: false,
            progress: null,
            progressMessage: ''
        };
        this.toggleModel = this.toggleModel.bind(this)
        this.detect = this.detect.bind(this)
        this.progressCallback = this.progressCallback.bind(this)
    }

    async detect() {
        this.setState({ predicting: true })
        let t0
        if (process.env.NODE_ENV === 'development') {
            t0 = performance.now();
        }
        const detections = await this.state.detector.detect(this.props.imageRef.current, 0.3)
        if (process.env.NODE_ENV === 'development') {
            var t1 = performance.now();
            console.log("Prediction took " + (t1 - t0) + " milliseconds.")
        }
        this.setState({ predicting: false })
        this.props.predictionCallback(detections)
    }

    progressCallback(progress, progressMessage) {
        this.setState({ progress, progressMessage })
    }

    async toggleModel() {
        this.setState({ togglingModel: true })
        let detector
        if (this.state.detector) {
            this.state.detector.dispose()
            detector = null
        } else {
            detector = await load(
                this.props.modelConfig.modelUrl,
                this.props.modelConfig.classes,
                this.props.modelConfig.preprocessingMode,
                this.progressCallback
            )
        }
        this.setState({ detector, togglingModel: false });
    }

    componentWillUnmount() {
        if (this.state.detector) {
            this.state.detector.dispose()
        }
    }

    render() {
        return (
            <div style={{ border: "2px dotted white", margin: "8px 0", padding: "0 8px 0px 8px" }}>
                <h2>{this.props.modelConfig.displayName}</h2>
                <div>
                    <button
                        style={{ width: "45%", marginRight: "5%", backgroundColor: "#EEEEEE" }}
                        variant="contained"
                        disabled={this.state.togglingModel}
                        onClick={this.toggleModel}>
                        {this.state.detector ? `Disable` : `Load (${this.props.modelConfig.modelSize})`}
                    </button>
                    <button
                        disabled={!this.state.detector || this.state.progress !== 1}
                        style={{ width: "45%", marginLeft: "5%" }}
                        onClick={this.detect}
                    >
                        Predict
                    </button>
                </div>
                <div>
                    {
                        (this.state.progress && this.state.progress !== 1) ?
                            <div style={{ marginTop: '10px' }}>
                                <Line percent={this.state.progress * 100} strokeWidth="4" strokeColor="#D3D3D3" />
                                <p style={{ textAlign: 'center', margin: 0 }}>{this.state.progressMessage}</p>
                            </div>
                            : null
                    }
                </div>
                <p>{this.props.modelConfig.description}</p>
            </div>
        );
    }
}
