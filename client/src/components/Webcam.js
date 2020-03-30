import React, { Component, useState } from 'react';
import { Button, Col, Row } from 'reactstrap';
import PageFrame from './PageFrame';
import * as handTrack from 'handtrackjs';
import Camera, { FACING_MODES, IMAGE_TYPES } from 'react-html5-camera-photo';
import 'react-html5-camera-photo/build/css/index.css';
import ImagePreview from './ImagePreview';
import axios from 'axios';

class Webcam extends Component {
    constructor(props) {
        super(props);

        this.state = {
            imguri: undefined,
            closeWebcam: false,
            isFullScreen: false,
            imgContent: undefined,
            loading: true,
        }
    }

    componentDidMount() {
        axios.get("http://localhost:8000/main/").then(() => {
            this.setState({loading: false});
        });
    }

    onCloseWebcam(e) {
        e.preventDefault();
        this.setState({closeWebcam: true});
    }

    handleTakePhoto(imguri) {
        this.setState({imguri: imguri, closeWebcam: true});
    }

    handleCancelImage(e) {
        e.preventDefault();
        this.setState({imguri: undefined, closeWebcam: false});
    }

    dataURItoBlob() {
        var byteString;
        var imguri = this.state.imguri;
        if (imguri.split(',')[0].indexOf('base64') >= 0)
            byteString = atob(imguri.split(',')[1]);
        else
            byteString = unescape(imguri.split(',')[1])
        
        var ia = new Uint8Array(byteString.length);
        for (var i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }

        return new Blob([ia], {type: 'image/jpeg'});
    }

    handleSubmitImage(e) {
        e.preventDefault();
        /*
        var blob = this.dataURItoBlob();
        console.log(blob);
        var fd = new FormData();
        fd.append("image", blob, "hand.jpeg");
        console.log(fd);
        */
        var postdata = {
            'uri': this.state.imguri
        }
        axios({
            method: 'post',
            url: "http://localhost:8000/main/",
            data: postdata
        }).then((res) => {
            console.log(res.data);
            this.setState({
                imgContent: res.data.content
            });
        });
    }

    render() {
        let content;
        if (!this.state.closeWebcam) {
            content = (
                <div>
                    <Camera
                        onTakePhotoAnimationDone={dataUri => {this.handleTakePhoto(dataUri)}}
                        isFullScreen={this.state.isFullScreen}
                        imageType = {IMAGE_TYPES.JPG}
                    />
                    <Button color="secondary" onClick={e => {this.onCloseWebcam(e)}}>Close Webcam</Button>
                </div>
            );
        } else if (this.state.imguri !== undefined) {
            content = <ImagePreview 
                        imguri={this.state.imguri}
                        handleCancelImage={e => {this.handleCancelImage(e)}}
                        handleSubmitImage={e => {this.handleSubmitImage(e)}} />
        }

        return (
            <PageFrame>
                <Row>
                    <Col>
                        {content}
                    </Col>
                    <Col>
                        <h5>{this.state.imgContent}</h5>
                    </Col>
                </Row>
            </PageFrame>
        );
    }
}

export default Webcam;