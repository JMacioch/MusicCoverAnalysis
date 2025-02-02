import "bootstrap/dist/css/bootstrap.min.css";
import "../styles/App.css";

import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import React, { useState } from "react";
import { createMemoryHistory } from "history";

/** import components */
import Main from "./Main";
import Survey from "./Survey";
import Result from "./Result";

function App() {
  const history = createMemoryHistory({ reducer: {} });
  const [surveyStarted, setSurveyStarted] = useState(false);
  const [surveyCompleted, setSurveyCompleted] = useState(false);

  function onStart() {
    setSurveyStarted(true);
  }

  function onFinish() {
    setSurveyCompleted(true);
  }

  return (
    <BrowserRouter location={history.location} navigator={history}>
      <Routes>
        <Route path="/" element={<Main onStart={onStart} />} />
        <Route
          path="/survey"
          element={
            surveyStarted ? <Survey onFinish={onFinish} /> : <Navigate to="/" />
          }
        />
        <Route
          path="/result"
          element={surveyCompleted ? <Result /> : <Navigate to="/survey" />}
        />
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
