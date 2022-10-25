//! A `MockSubscriber` to receive and validate `tracing` data.
//!
//! # Overview
//!
//! [`tracing`] is a framework for instrumenting Rust programs with context-aware,
//! structured, event-based diagnostic information. To test the `tracing` data
//! created by a crate, a `MockSubscriber` must be created. The `MockSubscriber`
//! can have expectations set on it which can then be validated by setting it as
//! the default subscriber.
//!
//! # Usage
//!
//! ```
//! use tracing::subscriber::with_default;
//! use tracing_mock::{event, field, span, subscriber};
//! 
//! let (subscriber, handle) = subscriber::mock()
//!        // Expect a single event with a specified message
//!        .event(event::mock().with_fields(field::msg("droids")))
//!        .done()
//!        .run_with_handle();
//! 
//! // Use `with_default` to apply the `MockSubscriber` for the duration
//! // of the closure - this is what we are testing.
//! with_default(subscriber, || {
//!     // These *are* the droids we are looking for
//!     tracing::info!("droids");
//! });
//! 
//! handle.assert_finished();
//! ```
#![allow(missing_docs)]
use super::{
    event::MockEvent,
    field as mock_field,
    span::{MockSpan, NewSpan},
};
use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
};
use tracing::{
    level_filters::LevelFilter,
    span::{self, Attributes, Id},
    subscriber::Interest,
    Event, Metadata, Subscriber,
};

/// FIXME: I want to see if we can make this non-public
#[derive(Debug, Eq, PartialEq)]
pub enum Expect {
    Event(MockEvent),
    FollowsFrom {
        consequence: MockSpan,
        cause: MockSpan,
    },
    Enter(MockSpan),
    Exit(MockSpan),
    CloneSpan(MockSpan),
    DropSpan(MockSpan),
    Visit(MockSpan, mock_field::Expect),
    NewSpan(NewSpan),
    Nothing,
}

struct SpanState {
    name: &'static str,
    refs: usize,
    meta: &'static Metadata<'static>,
}

struct Running<F: Fn(&Metadata<'_>) -> bool> {
    spans: Mutex<HashMap<Id, SpanState>>,
    expected: Arc<Mutex<VecDeque<Expect>>>,
    current: Mutex<Vec<Id>>,
    ids: AtomicUsize,
    max_level: Option<LevelFilter>,
    filter: F,
    name: String,
}

/// A subscriber which can validate received tracing data.
/// 
/// For a detailed description and examples see the documentation
/// for the methods and the [`subscriber`] module.
/// 
/// [`subscriber`]: mod@crate::subscriber
pub struct MockSubscriber<F: Fn(&Metadata<'_>) -> bool> {
    expected: VecDeque<Expect>,
    max_level: Option<LevelFilter>,
    filter: F,
    name: String,
}

/// A handle which is used to invoke validation of expectations.
pub struct MockHandle(Arc<Mutex<VecDeque<Expect>>>, String);

/// Create a new [`MockSubscriber`]
pub fn mock() -> MockSubscriber<fn(&Metadata<'_>) -> bool> {
    MockSubscriber {
        expected: VecDeque::new(),
        filter: (|_: &Metadata<'_>| true) as for<'r, 's> fn(&'r Metadata<'s>) -> _,
        max_level: None,
        name: thread::current()
            .name()
            .unwrap_or("mock_subscriber")
            .to_string(),
    }
}

impl<F> MockSubscriber<F>
where
    F: Fn(&Metadata<'_>) -> bool + 'static,
{
    /// Overrides the name printed by the mock subscriber's debugging output.
    ///
    /// The debugging output is displayed if the test panics, or if the test is
    /// run with `--nocapture`.
    ///
    /// By default, the mock subscriber's name is the  name of the test
    /// (*technically*, the name of the thread where it was created, which is
    /// the name of the test unless tests are run with `--test-threads=1`).
    /// When a test has only one mock subscriber, this is sufficient. However,
    /// some tests may include multiple subscribers, in order to test
    /// interactions between multiple subscribers. In that case, it can be
    /// helpful to give each subscriber a separate name to distinguish where the
    /// debugging output comes from.
    /// 
    /// ```
    /// use tracing::subscriber::with_default;
    /// use tracing_mock::{event, subscriber};
    /// 
    /// let (subscriber, handle) = subscriber::mock()
    ///     .named("subscriber-1")
    ///     .event(event::mock())
    ///     .done()
    ///     .run_with_handle();
    /// 
    /// with_default(subscriber, || {
    ///     tracing::info!("a");
    /// });
    /// 
    /// handle.assert_finished();
    /// ```
    pub fn named(self, name: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            ..self
        }
    }

    /// Expects a span matching the [`MockSpan`] to be entered.
    /// 
    /// This expectation is generally accompanied by a call to
    /// [`exit`] as well. If used together with [`done`], this
    /// is necessary.
    /// 
    /// ```
    /// use tracing::subscriber::with_default;
    /// use tracing_mock::{span, subscriber};
    /// 
    /// let span = span::mock()
    ///     .at_level(tracing::Level::INFO)
    ///     .named("the span we're testing");
    /// let (subscriber, handle) = subscriber::mock()
    ///     .enter(span.clone())
    ///     .exit(span)
    ///     .done()
    ///     .run_with_handle();
    /// 
    /// with_default(subscriber, || {
    ///     let span = tracing::info_span!("the span we're testing");
    ///     let _entered = span.enter();
    /// });
    /// 
    /// handle.assert_finished();
    /// ```
    /// 
    /// [`done`]: fn@Self::done
    /// [`exit`]: fn@Self::exit
    pub fn enter(mut self, span: MockSpan) -> Self {
        self.expected.push_back(Expect::Enter(span));
        self
    }

    /// Expects that a span matching `consequence` follows from a span matching `cause`.
    /// 
    /// For further details on what this causal relationship means, see
    /// [`Span::follows_from`].
    /// 
    /// ```
    /// use tracing::subscriber::with_default;
    /// use tracing_mock::{span, subscriber};
    /// 
    /// let span_1 = span::mock().named("cause");
    /// let span_2 = span::mock().named("consequence");
    /// 
    /// let (subscriber, handle) = subscriber::mock()
    ///     .new_span(span_1.clone())
    ///     .new_span(span_2.clone())
    ///     .follows_from(span_2, span_1)
    ///     .done()
    ///     .run_with_handle();
    /// 
    /// with_default(subscriber, || {
    ///     let cause = tracing::info_span!("cause");
    ///     let consequence = tracing::info_span!("consequence");
    /// 
    ///     consequence.follows_from(cause);
    /// });
    /// 
    /// handle.assert_finished();
    /// ```
    /// 
    /// [`Span::follows_from`]: fn@tracing::Span::follows_from
    pub fn follows_from(mut self, consequence: MockSpan, cause: MockSpan) -> Self {
        self.expected
            .push_back(Expect::FollowsFrom { consequence, cause });
        self
    }

    /// Expects an event matching the [`MockEvent`] to be traced.
    /// 
    /// The `event` can be simple a default mock which will match
    /// any event (`tracing-mock::event::mock()`) or can include
    /// additional requirements. See the [`MockEvent`] documentation
    /// for more details.
    /// 
    /// ```
    /// use tracing::subscriber::with_default;
    /// use tracing_mock::{event, subscriber};
    /// 
    /// let (subscriber, handle) = subscriber::mock()
    ///     .event(event::mock())
    ///     .done()
    ///     .run_with_handle();
    /// 
    /// with_default(subscriber, || {
    ///     tracing::info!("a");
    /// });
    /// 
    /// handle.assert_finished();
    /// ```    
    pub fn event(mut self, event: MockEvent) -> Self {
        self.expected.push_back(Expect::Event(event));
        self
    }

    /// Expects a span matching the [`MockSpan`] to exit.
    /// 
    /// As a span may be entered and exited multiple times,
    /// this is different from the span being dropped. In
    /// general [`enter`] and `exit` should be paired.
    /// 
    /// ```
    /// use tracing::subscriber::with_default;
    /// use tracing_mock::{span, subscriber};
    /// 
    /// let span = span::mock()
    ///     .at_level(tracing::Level::INFO)
    ///     .named("the span we're testing");
    /// let (subscriber, handle) = subscriber::mock()
    ///     .enter(span.clone())
    ///     .exit(span.clone())
    ///     .enter(span.clone())
    ///     .exit(span)
    ///     .done()
    ///     .run_with_handle();
    /// 
    /// with_default(subscriber, || {
    ///     let span = tracing::info_span!("the span we're testing");
    ///     {
    ///         let _entered = span.enter();
    ///     }
    ///     {
    ///         let _entered = span.enter();
    ///     }
    /// });
    /// 
    /// handle.assert_finished();
    /// ```
    /// 
    /// [`enter`]: fn@Self::enter
    pub fn exit(mut self, span: MockSpan) -> Self {
        self.expected.push_back(Expect::Exit(span));
        self
    }

    /// Expects a span matching the [`MockSpan`] to be cloned.
    /// 
    /// The cloned span does need to have been entered.
    /// 
    /// ```
    /// use tracing::subscriber::with_default;
    /// use tracing_mock::{span, subscriber};
    /// 
    /// let span = span::mock()
    ///     .at_level(tracing::Level::INFO)
    ///     .named("the span we're testing");
    /// let (subscriber, handle) = subscriber::mock()
    ///     .clone_span(span)
    ///     .done()
    ///     .run_with_handle();
    /// 
    /// with_default(subscriber, || {
    ///     let span = tracing::info_span!("the span we're testing");
    ///     _ = span.clone();
    /// });
    /// 
    /// handle.assert_finished();
    /// ```
    pub fn clone_span(mut self, span: MockSpan) -> Self {
        self.expected.push_back(Expect::CloneSpan(span));
        self
    }

    /// Expects a span matching the [`MockSpan`] to be dropped.
    /// 
    /// The cloned span does need to have been entered. This
    /// allows you to test that the span has its destructor run,
    /// as opposed to having `std::mem::forget` called on it.
    /// 
    /// ```
    /// use tracing::subscriber::with_default;
    /// use tracing_mock::{span, subscriber};
    /// 
    /// let span = span::mock()
    ///     .at_level(tracing::Level::INFO)
    ///     .named("the span we're testing");
    /// let (subscriber, handle) = subscriber::mock()
    ///     .drop_span(span)
    ///     .done()
    ///     .run_with_handle();
    /// 
    /// with_default(subscriber, || {
    ///     let _span = tracing::info_span!("the span we're testing");
    /// });
    /// 
    /// handle.assert_finished();
    /// ```
    #[allow(deprecated)]
    pub fn drop_span(mut self, span: MockSpan) -> Self {
        self.expected.push_back(Expect::DropSpan(span));
        self
    }

    /// Expects that no further traces are received.
    /// 
    /// Consider this simple test:
    /// 
    /// ```
    /// use tracing::subscriber::with_default;
    /// use tracing_mock::{event, subscriber};
    /// 
    /// let (subscriber, handle) = subscriber::mock()
    ///     .event(event::mock())
    ///     .done()
    ///     .run_with_handle();
    /// 
    /// with_default(subscriber, || {
    ///     tracing::info!("a");
    /// });
    /// 
    /// handle.assert_finished();
    /// ```
    /// 
    /// If, instead the call to `done` were removed, then additional
    /// events could be generated and the test would not fail.
    /// 
    /// ```
    /// use tracing::subscriber::with_default;
    /// use tracing_mock::{event, subscriber};
    /// 
    /// let (subscriber, handle) = subscriber::mock()
    ///     .event(event::mock())
    ///     //.done() <-- No call to done
    ///     .run_with_handle();
    /// 
    /// with_default(subscriber, || {
    ///     tracing::info!("a");
    ///     tracing::info!("a");
    ///     tracing::info!("a");
    /// });
    /// 
    /// handle.assert_finished();
    /// ```
    pub fn done(mut self) -> Self {
        self.expected.push_back(Expect::Nothing);
        self
    }

    /// Expect that `fields` are recorded on a span matching the [`MockSpan`].
    /// 
    /// 
    pub fn record<I>(mut self, span: MockSpan, fields: I) -> Self
    where
        I: Into<mock_field::Expect>,
    {
        self.expected.push_back(Expect::Visit(span, fields.into()));
        self
    }

    /// Expects a span matching `new_span` to be created.
    /// 
    /// This function accepts `Into<NewSpan>` instead of a [`MockSpan`]
    /// directly. So it can be used to test span fields and the span
    /// parent.
    /// 
    /// The new span doesn't need to have been entered.
    /// 
    /// ```
    /// use tracing::subscriber::with_default;
    /// use tracing_mock::{field, span, subscriber};
    /// 
    /// let span = span::mock()
    ///     .at_level(tracing::Level::INFO)
    ///     .named("the span we're testing")
    ///     .with_field(field::mock("testing").with_value(&"yes"));
    /// let (subscriber, handle) = subscriber::mock()
    ///     .new_span(span)
    ///     .done()
    ///     .run_with_handle();
    /// 
    /// with_default(subscriber, || {
    ///     _ = tracing::info_span!("the span we're testing", testing = "yes");
    /// });
    /// 
    /// handle.assert_finished();
    /// ```
    pub fn new_span<I>(mut self, new_span: I) -> Self
    where
        I: Into<NewSpan>,
    {
        self.expected.push_back(Expect::NewSpan(new_span.into()));
        self
    }

    /// Filter the traces evaluated by the `MockSubscriber`.
    /// 
    /// The filter will be applied to all traces received before
    /// any verification occurs - so its position in the call chain
    /// is not important.
    /// 
    /// ```
    /// use tracing::subscriber::with_default;
    /// use tracing_mock::{event, subscriber};
    /// 
    /// let (subscriber, handle) = subscriber::mock()
    ///     .with_filter(|meta| meta.level() <= &tracing::Level::WARN)
    ///     .event(event::mock())
    ///     .done()
    ///     .run_with_handle();
    /// 
    /// with_default(subscriber, || {
    ///     tracing::info!("a");
    ///     tracing::warn!("b");
    /// });
    /// 
    /// handle.assert_finished();
    /// ```
    pub fn with_filter<G>(self, filter: G) -> MockSubscriber<G>
    where
        G: Fn(&Metadata<'_>) -> bool + 'static,
    {
        MockSubscriber {
            expected: self.expected,
            filter,
            max_level: self.max_level,
            name: self.name,
        }
    }

    /// Sets the max level that will be provided to the `tracing` system.
    /// 
    /// This method can be used to test the internals of `tracing`, but it
    /// is also useful to filter out traces on more verbose levels if you
    /// only want to verify above a certain level.
    /// 
    /// ```
    /// use tracing::subscriber::with_default;
    /// use tracing_mock::{event, subscriber};
    /// 
    /// let (subscriber, handle) = subscriber::mock()
    ///     .with_max_level_hint(tracing::Level::INFO)
    ///     .event(event::mock().at_level(tracing::Level::INFO))
    ///     .done()
    ///     .run_with_handle();
    /// 
    /// with_default(subscriber, || {
    ///     tracing::debug!("a message we don't care about");
    ///     tracing::info!("a message we want to verify");
    /// });
    /// 
    /// handle.assert_finished();
    /// ```
    /// 
    /// Note that this value determines a global filter, if `with_max_level_hint`
    /// is called on multiple subscribers, the global filter will be the least
    /// restrictive of all subscribers.
    /// 
    /// To filter the events evaluated by a specific `MockSubscriebr`, use
    /// [`with_filter`] instead.
    /// 
    /// [`with_filter`]: fn@Self::with_filter
    pub fn with_max_level_hint(self, hint: impl Into<LevelFilter>) -> Self {
        Self {
            max_level: Some(hint.into()),
            ..self
        }
    }

    pub fn run(self) -> impl Subscriber {
        let (subscriber, _) = self.run_with_handle();
        subscriber
    }

    pub fn run_with_handle(self) -> (impl Subscriber, MockHandle) {
        let expected = Arc::new(Mutex::new(self.expected));
        let handle = MockHandle(expected.clone(), self.name.clone());
        let subscriber = Running {
            spans: Mutex::new(HashMap::new()),
            expected,
            current: Mutex::new(Vec::new()),
            ids: AtomicUsize::new(1),
            filter: self.filter,
            max_level: self.max_level,
            name: self.name,
        };
        (subscriber, handle)
    }
}

impl<F> Subscriber for Running<F>
where
    F: Fn(&Metadata<'_>) -> bool + 'static,
{
    fn enabled(&self, meta: &Metadata<'_>) -> bool {
        println!("[{}] enabled: {:#?}", self.name, meta);
        let enabled = (self.filter)(meta);
        println!("[{}] enabled -> {}", self.name, enabled);
        enabled
    }

    fn register_callsite(&self, meta: &'static Metadata<'static>) -> Interest {
        println!("[{}] register_callsite: {:#?}", self.name, meta);
        if self.enabled(meta) {
            Interest::always()
        } else {
            Interest::never()
        }
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        println!("[{}] max_level_hint ->  {:?}", self.name, self.max_level);
        self.max_level
    }

    fn record(&self, id: &Id, values: &span::Record<'_>) {
        let spans = self.spans.lock().unwrap();
        let mut expected = self.expected.lock().unwrap();
        let span = spans
            .get(id)
            .unwrap_or_else(|| panic!("[{}] no span for ID {:?}", self.name, id));
        println!(
            "[{}] record: {}; id={:?}; values={:?};",
            self.name, span.name, id, values
        );
        let was_expected = matches!(expected.front(), Some(Expect::Visit(_, _)));
        if was_expected {
            if let Expect::Visit(expected_span, mut expected_values) = expected.pop_front().unwrap()
            {
                if let Some(name) = expected_span.name() {
                    assert_eq!(name, span.name);
                }
                let context = format!("span {}: ", span.name);
                let mut checker = expected_values.checker(&context, &self.name);
                values.record(&mut checker);
                checker.finish();
            }
        }
    }

    fn event(&self, event: &Event<'_>) {
        let name = event.metadata().name();
        println!("[{}] event: {};", self.name, name);
        match self.expected.lock().unwrap().pop_front() {
            None => {}
            Some(Expect::Event(mut expected)) => {
                let get_parent_name = || {
                    let stack = self.current.lock().unwrap();
                    let spans = self.spans.lock().unwrap();
                    event
                        .parent()
                        .and_then(|id| spans.get(id))
                        .or_else(|| stack.last().and_then(|id| spans.get(id)))
                        .map(|s| s.name.to_string())
                };
                expected.check(event, get_parent_name, &self.name);
            }
            Some(ex) => ex.bad(&self.name, format_args!("observed event {:#?}", event)),
        }
    }

    fn record_follows_from(&self, consequence_id: &Id, cause_id: &Id) {
        let spans = self.spans.lock().unwrap();
        if let Some(consequence_span) = spans.get(consequence_id) {
            if let Some(cause_span) = spans.get(cause_id) {
                println!(
                    "[{}] record_follows_from: {} (id={:?}) follows {} (id={:?})",
                    self.name, consequence_span.name, consequence_id, cause_span.name, cause_id,
                );
                match self.expected.lock().unwrap().pop_front() {
                    None => {}
                    Some(Expect::FollowsFrom {
                        consequence: ref expected_consequence,
                        cause: ref expected_cause,
                    }) => {
                        if let Some(name) = expected_consequence.name() {
                            assert_eq!(name, consequence_span.name);
                        }
                        if let Some(name) = expected_cause.name() {
                            assert_eq!(name, cause_span.name);
                        }
                    }
                    Some(ex) => ex.bad(
                        &self.name,
                        format_args!(
                            "consequence {:?} followed cause {:?}",
                            consequence_span.name, cause_span.name
                        ),
                    ),
                }
            }
        };
    }

    fn new_span(&self, span: &Attributes<'_>) -> Id {
        let meta = span.metadata();
        let id = self.ids.fetch_add(1, Ordering::SeqCst);
        let id = Id::from_u64(id as u64);
        println!(
            "[{}] new_span: name={:?}; target={:?}; id={:?};",
            self.name,
            meta.name(),
            meta.target(),
            id
        );
        let mut expected = self.expected.lock().unwrap();
        let was_expected = matches!(expected.front(), Some(Expect::NewSpan(_)));
        let mut spans = self.spans.lock().unwrap();
        if was_expected {
            if let Expect::NewSpan(mut expected) = expected.pop_front().unwrap() {
                let get_parent_name = || {
                    let stack = self.current.lock().unwrap();
                    span.parent()
                        .and_then(|id| spans.get(id))
                        .or_else(|| stack.last().and_then(|id| spans.get(id)))
                        .map(|s| s.name.to_string())
                };
                expected.check(span, get_parent_name, &self.name);
            }
        }
        spans.insert(
            id.clone(),
            SpanState {
                name: meta.name(),
                meta,
                refs: 1,
            },
        );
        id
    }

    fn enter(&self, id: &Id) {
        let spans = self.spans.lock().unwrap();
        if let Some(span) = spans.get(id) {
            println!("[{}] enter: {}; id={:?};", self.name, span.name, id);
            match self.expected.lock().unwrap().pop_front() {
                None => {}
                Some(Expect::Enter(ref expected_span)) => {
                    if let Some(name) = expected_span.name() {
                        assert_eq!(name, span.name);
                    }
                }
                Some(ex) => ex.bad(&self.name, format_args!("entered span {:?}", span.name)),
            }
        };
        self.current.lock().unwrap().push(id.clone());
    }

    fn exit(&self, id: &Id) {
        if std::thread::panicking() {
            // `exit()` can be called in `drop` impls, so we must guard against
            // double panics.
            println!("[{}] exit {:?} while panicking", self.name, id);
            return;
        }
        let spans = self.spans.lock().unwrap();
        let span = spans
            .get(id)
            .unwrap_or_else(|| panic!("[{}] no span for ID {:?}", self.name, id));
        println!("[{}] exit: {}; id={:?};", self.name, span.name, id);
        match self.expected.lock().unwrap().pop_front() {
            None => {}
            Some(Expect::Exit(ref expected_span)) => {
                if let Some(name) = expected_span.name() {
                    assert_eq!(name, span.name);
                }
                let curr = self.current.lock().unwrap().pop();
                assert_eq!(
                    Some(id),
                    curr.as_ref(),
                    "[{}] exited span {:?}, but the current span was {:?}",
                    self.name,
                    span.name,
                    curr.as_ref().and_then(|id| spans.get(id)).map(|s| s.name)
                );
            }
            Some(ex) => ex.bad(&self.name, format_args!("exited span {:?}", span.name)),
        };
    }

    fn clone_span(&self, id: &Id) -> Id {
        let name = self.spans.lock().unwrap().get_mut(id).map(|span| {
            let name = span.name;
            println!(
                "[{}] clone_span: {}; id={:?}; refs={:?};",
                self.name, name, id, span.refs
            );
            span.refs += 1;
            name
        });
        if name.is_none() {
            println!("[{}] clone_span: id={:?};", self.name, id);
        }
        let mut expected = self.expected.lock().unwrap();
        let was_expected = if let Some(Expect::CloneSpan(ref span)) = expected.front() {
            assert_eq!(
                name,
                span.name(),
                "[{}] expected to clone a span named {:?}",
                self.name,
                span.name()
            );
            true
        } else {
            false
        };
        if was_expected {
            expected.pop_front();
        }
        id.clone()
    }

    fn drop_span(&self, id: Id) {
        let mut is_event = false;
        let name = if let Ok(mut spans) = self.spans.try_lock() {
            spans.get_mut(&id).map(|span| {
                let name = span.name;
                if name.contains("event") {
                    is_event = true;
                }
                println!(
                    "[{}] drop_span: {}; id={:?}; refs={:?};",
                    self.name, name, id, span.refs
                );
                span.refs -= 1;
                name
            })
        } else {
            None
        };
        if name.is_none() {
            println!("[{}] drop_span: id={:?}", self.name, id);
        }
        if let Ok(mut expected) = self.expected.try_lock() {
            let was_expected = match expected.front() {
                Some(Expect::DropSpan(ref span)) => {
                    // Don't assert if this function was called while panicking,
                    // as failing the assertion can cause a double panic.
                    if !::std::thread::panicking() {
                        assert_eq!(name, span.name());
                    }
                    true
                }
                Some(Expect::Event(_)) => {
                    if !::std::thread::panicking() {
                        assert!(is_event, "[{}] expected an event", self.name);
                    }
                    true
                }
                _ => false,
            };
            if was_expected {
                expected.pop_front();
            }
        }
    }

    fn current_span(&self) -> tracing_core::span::Current {
        let stack = self.current.lock().unwrap();
        match stack.last() {
            Some(id) => {
                let spans = self.spans.lock().unwrap();
                let state = spans.get(id).expect("state for current span");
                tracing_core::span::Current::new(id.clone(), state.meta)
            }
            None => tracing_core::span::Current::none(),
        }
    }
}

impl MockHandle {
    /// TODO: I want to see if we can make this non-public.
    pub fn new(expected: Arc<Mutex<VecDeque<Expect>>>, name: String) -> Self {
        Self(expected, name)
    }

    /// Check all of the expectations specified on the associated [`MockSubscriber`].
    /// 
    /// Calling `assert_finished` is usually the final part of a test.
    /// 
    /// # Panics
    /// 
    /// This method will panic if any of the provided expectations are not met.
    ///
    /// ```
    /// use tracing::subscriber::with_default;
    /// use tracing_mock::{event, subscriber};
    /// 
    /// let (subscriber, handle) = subscriber::mock()
    ///     .event(event::mock())
    ///     .done()
    ///     .run_with_handle();
    /// 
    /// with_default(subscriber, || {
    ///     tracing::info!("a");
    /// });
    /// 
    /// handle.assert_finished();
    /// ```   
    pub fn assert_finished(&self) {
        if let Ok(ref expected) = self.0.lock() {
            assert!(
                !expected.iter().any(|thing| thing != &Expect::Nothing),
                "\n[{}] more notifications expected: {:#?}",
                self.1,
                **expected
            );
        }
    }
}

impl Expect {
    pub fn bad(&self, name: impl AsRef<str>, what: fmt::Arguments<'_>) {
        let name = name.as_ref();
        match self {
            Expect::Event(e) => panic!(
                "\n[{}] expected event {}\n[{}] but instead {}",
                name, e, name, what,
            ),
            Expect::FollowsFrom { consequence, cause } => panic!(
                "\n[{}] expected consequence {} to follow cause {} but instead {}",
                name, consequence, cause, what,
            ),
            Expect::Enter(e) => panic!(
                "\n[{}] expected to enter {}\n[{}] but instead {}",
                name, e, name, what,
            ),
            Expect::Exit(e) => panic!(
                "\n[{}] expected to exit {}\n[{}] but instead {}",
                name, e, name, what,
            ),
            Expect::CloneSpan(e) => {
                panic!(
                    "\n[{}] expected to clone {}\n[{}] but instead {}",
                    name, e, name, what,
                )
            }
            Expect::DropSpan(e) => {
                panic!(
                    "\n[{}] expected to drop {}\n[{}] but instead {}",
                    name, e, name, what,
                )
            }
            Expect::Visit(e, fields) => panic!(
                "\n[{}] expected {} to record {}\n[{}] but instead {}",
                name, e, fields, name, what,
            ),
            Expect::NewSpan(e) => panic!(
                "\n[{}] expected {}\n[{}] but instead {}",
                name, e, name, what
            ),
            Expect::Nothing => panic!(
                "\n[{}] expected nothing else to happen\n[{}] but {} instead",
                name, name, what,
            ),
        }
    }
}
