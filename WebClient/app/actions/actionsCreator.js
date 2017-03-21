import constants from '../constants';
import commonApi from '../api/common';

let actionsCreator = {

  uploadFiles(files) {
    return (dispatch) => {
        dispatch({
            type: constants.UPLOADING_STARTED
        });
        commonApi.uploadFiles(files)
            .then((result) => {
                dispatch({
                type: constants.VIDEO_UPLOADED,
                filePath: result.body.filePath
              });
              dispatch(actionsCreator.getTimeStamps(result.body.fileName));
            })
    }
  },

  getTimeStamps(fileName){
      return (dispatch) => {
          commonApi.getTimeStamps(fileName)
              .then((timeStamps) => {
                  dispatch({
                      type: constants.TIME_STAMPS_RESEVED,
                      timeStamps: timeStamps
                  });
              })
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
