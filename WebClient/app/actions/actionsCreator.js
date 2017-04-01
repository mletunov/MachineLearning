import constants from '../constants';
import commonApi from '../api/common';
import { browserHistory } from 'react-router';
import {logSrv} from '../helpers';

let actionsCreator = {

  videoPageOpened(){
    return {
      type: constants.UPLOAD_PAGE_OPENED
    }
  },
  uploadFiles(files) {
    return (dispatch) => {
        dispatch({
            type: constants.UPLOADING_STARTED
        });
        commonApi.uploadFiles(files)
            .then((result) => {
                browserHistory.push(`/deeplink/${result.body.session}`)
            }).catch(() =>{
              logSrv.Error();
              dispatch({
                type:constants.ON_ERROR
              })
            })
    }
  },

  getSessionData(sessionId){
      return (dispatch) => {


          var callBack = function(dispatch){
            commonApi.getSessionData(sessionId)
                .then((result) => {
                    if(result.video && result.time){
                      dispatch({
                          type: constants.ALL_PROCESSED,
                          timeStamps: result.time,
                          filePath: result.video
                      });

                    } else if (result.video){
                      dispatch({
                          type: constants.VIDEO_PROCESSED,
                          filePath: result.video
                      });
                      setTimeout(function(){ callBack(dispatch);}, 2000);
                    } else if (result.time){
                      dispatch({
                          type: constants.STAMPS_PROCESSED,
                          timeStamps: result.time
                      })
                      setTimeout(function(){ callBack(dispatch);}, 2000);
                    } else {
                      setTimeout(function(){ callBack(dispatch);}, 2000);
                    }
                  }).catch(() =>{
                    logSrv.Error();
                    dispatch({
                      type:constants.ON_ERROR
                    })
                  })
          };

          callBack(dispatch);
      }

  },

  seekVideo(time) {
      return {
          type: constants.SEEK_VIDEO,
          time: time
      }
  }
};

export default actionsCreator;
