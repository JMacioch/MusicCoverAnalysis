export default [
  {
    id: 1,
    question:
      "Please pick a color that best represents the dominant color of the provided album cover art",
    image:
      "https://lastfm.freetls.fastly.net/i/u/300x300/b6e6b6af5aa815e18e3a149464d0fe41.jpg",
    color: true,
  },
  {
    id: 2,
    question:
      "What music genre would you assign this album judging by the provided album cover art?",
    image:
      "https://lastfm.freetls.fastly.net/i/u/300x300/ef4c66773dd7aae88ed64550ce55bb5a.jpg",
    options: [
      "blues",
      "classical",
      "country",
      "electronic",
      "hip-hop",
      "jazz",
      "metal",
      "pop",
      "reggae",
      "rock",
    ],
    color: false,
  },
  {
    id: 3,
    question: "What music genre would you assign this color?",
    image: "https://i.postimg.cc/dVJPPbjc/solid-color-image-02.jpg",
    options: [
      "blues",
      "classical",
      "country",
      "electronic",
      "hip-hop",
      "jazz",
      "metal",
      "pop",
      "reggae",
      "rock",
    ],
    color: false,
  },
];
