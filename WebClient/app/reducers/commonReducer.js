import constants from '../constants';
import update from 'react-addons-update';

let initialState = {
  filePath: '',
  timeStamps: [],
  timeSeek: 0
};
const initialAction = { type: 'initial state'}

const common = (state = initialState, action = initialAction) => {

    switch (action.type) {
        case constants.VIDEO_PROCESSED:
            return update(state, {
                filePath: { $set: `${constants.API}${action.filePath}` }
            });
        case constants.STAMPS_PROCESSED:
            return update(state, {
                timeStamps: { $set: action.timeStamps }
            });
        case constants.ALL_PROCESSED:
            return update(state, {
                timeStamps: { $set: action.timeStamps },
                filePath: { $set: `${constants.API_FOR_FILE}${action.filePath.replace(/\\/g, "/")}` }
            });

        case constants.SEEK_VIDEO:
            return update(state, {
                timeSeek: { $set: action.time }
            });

        default:
          return state;
    }
}

export default common;
