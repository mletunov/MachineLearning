import React, { Component } from 'react';
import request from 'superagent';
import Loader from 'react-loader';
import Header from './header';
import Footer from './footer';
import Uploader from './uploader'
import TimeStamps from './stamps'
import Video from './video'
import actions from '../actions/actionsCreator'
import { connect } from 'react-redux'

class Main extends Component {

    constructor(props) {
        super(props);
    }


    onFileSelected(files){
        this.props.uploadVideo(files);
    }

    onTimeSeek(time) {
        this.props.seekVideo(time);
    }

    render() {
        return (
            <div>
                <Header/>
                <div className="container content-block">

                        <div className="col-md-3">
                            <Uploader onFileSelected={this.onFileSelected.bind(this)} />
                        </div>
                        <div className="col-md-8">
                            <Loader loaded={!this.props.isLoading} className={this.props.isLoading ? "loader-spin" : ""}>
                                <Video filePath={this.props.filePath} timeSeek={this.props.timeSeek} />
                                <TimeStamps stamps={this.props.timeStamps} onTimeSeek={this.onTimeSeek.bind(this)} />
                            </Loader>
                        </div>
                </div>
                <Footer/>
            </div>
        );
    }
}




function mapStoreToProps(storeState) {
  return {
    timeStamps: storeState.commonReducer.timeStamps,
    filePath: storeState.commonReducer.filePath,
    timeSeek: storeState.commonReducer.timeSeek,
    isLoading: storeState.commonReducer.isLoading
  }
}

function mapDispatchToProps(dispatch) {
  return {
      uploadVideo: (files) => dispatch(actions.uploadFiles(files)),
      seekVideo: (time) => dispatch(actions.seekVideo(time))
  }
}

export default connect(
  mapStoreToProps,
  mapDispatchToProps)
(Main)
