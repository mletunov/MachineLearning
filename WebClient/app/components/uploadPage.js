import React, { Component } from 'react';
import Uploader from './uploader';
import { connect } from 'react-redux';
import Loading from './loading';
import actions from '../actions/actionsCreator';

class UploadPage extends Component {

    constructor(props) {
        super(props);
    }


    onFileSelected(files){
        this.props.uploadVideo(files);
    }
    componentWillMount(){
      this.props.videoPageOpened();
    }

    render() {
        return (
            <div className="upl-pg">
              <Loading isLoading={this.props.isLoading} isBig={true}>
                <Uploader onFileSelected={this.onFileSelected.bind(this)} />
              </Loading>
            </div>
        );
    }
}


function mapStoreToProps(storeState) {
  return {
    isLoading: storeState.loadingReducer.pageLoading,
    filePath: storeState.commonReducer.filePath,
    timeSeek: storeState.commonReducer.timeSeek
  }
}

function mapDispatchToProps(dispatch) {
  return {
      uploadVideo: (files) => dispatch(actions.uploadFiles(files)),
      videoPageOpened: (files) => dispatch(actions.videoPageOpened(files)),
  }
}

export default connect(
  mapStoreToProps,
  mapDispatchToProps)
(UploadPage)
