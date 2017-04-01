import React, { Component } from 'react';
import Uploader from './uploader';
import { connect } from 'react-redux';
import Video from './video';
import Stamps from './stamps';
import Loading from './loading';
import actions from '../actions/actionsCreator';
import { browserHistory } from 'react-router';

class VideoPage extends Component {

    constructor(props) {
        super(props);
    }


    onTimeSeek(time) {
        this.props.seekVideo(time);
    }

    componentDidMount(){
      this.props.getSessionData(this.props.params.sessionId);
    }

    startNew(){
      browserHistory.push("/");
    }

    render() {
        return (
            <div className="video-pg">
                <div className="video-pg-head">
                  <button onClick={() => this.startNew()} className="btn btn-primary">Start New</button>
                  <div className="input-group input-group-sm video-link">
                    <span className="input-group-addon" id="sizing-addon3">Link</span>
                    <input type="text" className="form-control" placeholder="Link" aria-describedby="sizing-addon3" onChange={()=>{}} value={window.location.href}/>
                  </div>
                </div>
                <div className="video-blk">
                  <Loading isLoading={this.props.videoLoading} isBig={false}>
                    <Video filePath={this.props.filePath} timeSeek={this.props.timeSeek} />
                  </Loading>
                </div>
                <div className="stamps-blk">
                  <Loading isLoading={this.props.stampsLoading} isBig={false}>
                    <Stamps stamps={this.props.timeStamps} onTimeSeek={this.onTimeSeek.bind(this)} />
                  </Loading>
                </div>
            </div>
        );
    }
}




function mapStoreToProps(storeState) {
  return {
    timeStamps: storeState.commonReducer.timeStamps,
    filePath: storeState.commonReducer.filePath,
    timeSeek: storeState.commonReducer.timeSeek,
    videoLoading: storeState.loadingReducer.videoLoading,
    stampsLoading: storeState.loadingReducer.stampsLoading,
  }
}

function mapDispatchToProps(dispatch) {
  return {
      seekVideo: (time) => dispatch(actions.seekVideo(time)),
      getSessionData: (sessionId) => dispatch(actions.getSessionData(sessionId))
  }
}

export default connect(
  mapStoreToProps,
  mapDispatchToProps)
(VideoPage)
