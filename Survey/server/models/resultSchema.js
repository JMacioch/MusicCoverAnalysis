import mongoose from "mongoose";
const { Schema } = mongoose;

/** result model */
const resultModel = new Schema({
  result: { type: Array, default: [] },
  createdAt: { type: Date, default: Date.now },
});

export default mongoose.model("Result", resultModel);
