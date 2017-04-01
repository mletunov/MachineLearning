import constants from '../constants';
import update from 'react-addons-update';

let initialState = {
  pageLoading: false,
  videoLoading: true,
  stampsLoading: true
};
const initialAction = { type: 'initial state'}

const loading = (state = initialState, action = initialAction) => {
    switch (action.type) {
        case constants.UPLOADING_STARTED:
            return update(state, {
                pageLoading: { $set: true },
                stampsLoading: { $set: true },
                videoLoading: { $set: true }
            });
        case constants.VIDEO_PROCESSED:
            return update(state, {
                videoLoading: { $set: false }
            });
        case constants.STAMPS_PROCESSED:
            return update(state, {
                stampsLoading: { $set: false }
            });
        case constants.ALL_PROCESSED:
            return update(state, {
                stampsLoading: { $set: false },
                videoLoading: { $set: false }
            });
        case constants.UPLOAD_PAGE_OPENED:
          return update(state, {
              pageLoading: { $set: false },
              stampsLoading: { $set: true },
              videoLoading: { $set: true }
          });
          case constants.ON_ERROR:
            return update(state, {
                pageLoading: { $set: false },
                stampsLoading: { $set: false },
                videoLoading: { $set: false }
            });
        default:
          return state;
    }
}

export default loading;
