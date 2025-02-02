import Questions from "../models/questionSchema.js";
import Result from "../models/resultSchema.js";
import questions from "../database/data.js";

/** get all questions */
export async function getQuestions(req, res) {
  try {
    const q = await Questions.find();
    res.json(q);
  } catch (error) {
    res.json({ error });
  }
}

/** insert all questions */
export async function insertQuestions(req, res) {
  try {
    Questions.insertMany({ questions }).then(function () {
      res.json({ msg: "Successfully saved defult items to DB" });
    });
  } catch (error) {
    res.json({ error });
  }
}

/** delete all questions */
export async function dropQuestions(req, res) {
  try {
    await Questions.deleteMany();
    res.json({ msg: "Successfully deleted items from DB" });
  } catch (error) {
    res.json({ error });
  }
}

/** get all result */
export async function getResult(req, res) {
  try {
    const r = await Result.find();
    res.json(r);
  } catch (error) {
    res.json({ error });
  }
}

/** insert all result */
export async function insertResult(req, res) {
  try {
    const { result } = req.body;
    if (!result) throw new Error("Data not provided!");
    Result.create({ result }).then(function () {
      res.json({ msg: "Successfully saved items to DB" });
    });
  } catch (error) {
    res.json({ error });
  }
}

/** delete all result */
export async function dropResult(req, res) {
  try {
    await Result.deleteMany();
    res.json({ msg: "Successfully deleted items from DB" });
  } catch (error) {
    res.json({ error });
  }
}
