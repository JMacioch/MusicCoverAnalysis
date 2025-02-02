import React, { useEffect, useState } from "react";
import { MoveNextQuestion, MovePrevQuestion } from "../hooks/FetchQuestions.js";
import { PushAnswer } from "../hooks/SetResult.js";
import { Link, Navigate } from "react-router-dom";
import { updateResult } from "../hooks/SetResult.js";
/** custom hook */
import { useFetchQuestion } from "../hooks/FetchQuestions.js";
import { SketchPicker } from "react-color";
import {
  Navbar,
  Container,
  Row,
  Col,
  Form,
  Button,
  Image,
  FormLabel,
} from "react-bootstrap";

/** redux store import */
import { useSelector, useDispatch } from "react-redux";

export default function Survey({ onFinish }) {
  const [checked, setChecked] = useState(undefined);
  const [{ isLoading, apiData, serverError }] = useFetchQuestion();
  const [color, setColor] = useState("#ff0000"); // Default color set to red
  const result = useSelector((state) => state.result.result);
  const { queue, trace } = useSelector((state) => state.questions);
  const [warning, setWarning] = useState("");
  const dispatch = useDispatch();

  useEffect(() => {
    // Set the checked state to the existing answer if available
    if (result[trace] !== undefined) {
      setChecked(result[trace]);
    } else {
      setChecked(undefined);
    }
  }, [trace, result]);

  const handleChangeComplete = (newColor) => {
    setColor(newColor.hex); // Updates state with the new color
    onChecked(newColor.hex);
    setChecked(newColor.hex);
    dispatch(updateResult({ trace, checked }));
  };

  const questions = useSelector(
    (state) => state.questions.queue[state.questions.trace]
  );

  const picker = document.getElementById("picker");
  const open = (e) => {
    // OPEN Callback
    picker.setAttribute("open", true);
  };

  useEffect(() => {
    dispatch(updateResult({ trace, checked }));
  }, [checked]);

  function onSelect(i) {
    onChecked(i);
    setChecked(i);
    dispatch(updateResult({ trace, checked }));
  }

  if (isLoading)
    return <h3 className="text-light">Content of the page is loading...</h3>;
  if (serverError)
    return (
      <h3 className="text-light">
        {serverError.toString() || "Unknown Error"}
      </h3>
    );

  /** next button event handler */
  function onNext() {
    if (checked === undefined) {
      setWarning("Please select an answer before proceeding.");
      return;
    }

    setWarning(""); // Clear any previous warning message

    if (trace < queue.length) {
      /** increase the trace value by one using MoveNextAction*/
      dispatch(MoveNextQuestion());
      /** insert a new result in the array */
      if (result.length <= trace) {
        dispatch(PushAnswer(checked));
      }
    }

    /** reset the value of the checked variable */
    setChecked(undefined);
  }

  /** previous button event handler */
  function onPrev() {
    if (trace > 0) {
      /** decrease the trace value by one using MovePrevAction*/
      dispatch(MovePrevQuestion());
    }
    setWarning("");
  }

  function onChecked(checked) {
    setChecked(checked);
    setWarning(""); // Clear any warning when the user selects an answer
  }

  function onFinished() {
    if (checked === undefined) {
      setWarning("Please select an answer before finishing.");
      return;
    }
    onNext();
    onFinish();
  }

  /** finished exam after the last question */
  if (result.length && result.length >= queue.length) {
    return <Navigate to={"/result"} replace="true"></Navigate>;
  }

  return (
    <div>
      {/* Navbar Section */}
      <Navbar bg="dark" variant="dark" expand={false}>
        <Container>
          <Navbar.Brand>
            <Link class="navbar-brand" to="/">
              Survey Application
            </Link>
          </Navbar.Brand>
        </Container>
      </Navbar>

      <Container className="container-body mt-5 justify-content-center align-items-center vh-50">
        <Row className="align-items-center mt-5 mb-5 ms-5 me-5">
          <Col xs={12} md={8} className="order-md-1 order-2 mb-4 mb-md-0">
            <Form>
              <FormLabel>
                <strong>{questions?.question}</strong>
              </FormLabel>
              <Row className="mt-2">
                {questions?.color ? (
                  <div className="d-flex justify-content-center">
                    <SketchPicker
                      color={result[trace] ? result[trace] : color}
                      onChangeComplete={handleChangeComplete}
                    />
                  </div>
                ) : (
                  <>
                    {questions?.options.map((q, i) => (
                      <Col xs={12} sm={6} key={i}>
                        <Form.Check
                          type="radio"
                          id={`q${i}-option`}
                          name="satisfaction"
                          value={q}
                          label={q}
                          onChange={() => onSelect(i)}
                          checked={checked === i}
                          className="mb-2"
                        />
                      </Col>
                    ))}
                  </>
                )}
              </Row>
              <div className="d-flex justify-content-between mt-4">
                {trace > 0 ? (
                  <Button variant="secondary" onClick={onPrev}>
                    Previous
                  </Button>
                ) : (
                  <div></div>
                )}
                {warning && <div className="warning">{warning}</div>}
                {trace === queue.length - 1 ? (
                  <Button variant="primary" onClick={onFinished}>
                    Finish
                  </Button>
                ) : (
                  <Button variant="primary" onClick={onNext}>
                    Next
                  </Button>
                )}
              </div>
            </Form>
          </Col>
          <Col xs={12} md={4} className="text-center order-md-2 order-1">
            <img src={questions?.image} className="img-fluid mb-3 mb-md-0" />
          </Col>
        </Row>
      </Container>

      {/* Footer Section */}
      <footer className="mt-5 mb-5 text-center text-muted">
        &copy; 2024 Survey Application. All rights reserved.
      </footer>
    </div>
  );
}
